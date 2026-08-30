"""batch_decode -- gom LO o buoc DECODE tu cache da dung san.

User 2026-08-30: "eval hien tai qua lau" + "sao ko dung ky thuat cua vllm,
cat phan can bê qua".

VI SAO KHONG PHAI FLASH-ATTENTION. Do tren chinh cau hinh nay: decode mot
token tren 9B, ngu canh ~500 -> nhan trong so ~18 GFLOP, attention ~6 MFLOP.
Attention chiem 0,03%. Nut co chai la BATCH = 1: moi token phai doc ~5GB
trong so 4-bit tu HBM (tran ly thuyet ~60 tok/s tren L4; dang duoc 10 tok/s).
Gom lo lam mot lan doc trong so phuc vu B mau -> danh thang vao nut co chai.

VI SAO GOM LO O DECODE AN TOAN, TRONG KHI O PREFILL THI KHONG.
probe_batch da do: dem trai lam attention keys sai 96% (lech vi tri RoPE) va
lam trang thai GDN xau hon nen 3,7 lan (vong quet hoi quy nuot token dem).
Nhung ca hai loi do thuoc ve PREFILL:
  - GDN: trang thai la ma tran (B, H, dk, dv) — KHONG CO chieu thoi gian.
    Cache da dung xong roi thi xep chong la chuyen tam thuong, khong dem gi.
  - RoPE: KV trong cache DA duoc ap RoPE theo vi tri THAT cua tung mau. Dem
    trai chi doi CHI SO trong tensor, khong doi vi tri da nhung. Token moi
    thi ta truyen position_ids RIENG cho tung hang (= do dai that cua hang
    do), nen khong bi keo theo T_max.
Ta TU DUNG cache chu khong di qua forward, nen dat duoc mask va position_ids
theo tung hang — dung cai ma prefill khong cho phep.

BAT BUOC co cong kiem: chay cung mot bo mau o batch 1 va batch B roi doi
chieu. Khong co no thi mot loi mask/vi tri se lam sai TOAN BO so bao cao ma
khong nem loi nao.
"""

import importlib.util
from pathlib import Path

import torch

_H = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e5 = _load("e5_train")


def stack_students(pasts):
    """Gop nhieu cache (moi cai batch 1) thanh MOT cache batch B.

    Tra ve (past_batched, lens) voi lens[i] = do dai that cua hang i.
    Attention KV duoc DEM TRAI bang 0 den T_max; GDN xep chong thang.
    """
    B = len(pasts)
    lens = []
    a0, _ = e5.split_layers(pasts[0])
    k_any = e5._get(a0[sorted(a0)[0]].keys)
    for p in pasts:
        a, _ = e5.split_layers(p)
        lens.append(e5._get(a[sorted(a)[0]].keys).shape[2])
    T_max = max(lens)

    out = e5.clone_cache_struct(pasts[0])
    attn_o, gdn_o = e5.split_layers(out)
    for j in sorted(attn_o):
        ks, vs = [], []
        for p in pasts:
            a, _ = e5.split_layers(p)
            k = e5._get(a[j].keys)
            v = e5._get(a[j].values)
            pad = T_max - k.shape[2]
            if pad:
                z = torch.zeros(k.shape[0], k.shape[1], pad, k.shape[3],
                                dtype=k.dtype, device=k.device)
                k = torch.cat([z, k], 2)      # DEM TRAI
                v = torch.cat([z, v], 2)
            ks.append(k)
            vs.append(v)
        e5._set_like(attn_o[j], "keys", torch.cat(ks, 0))
        e5._set_like(attn_o[j], "values", torch.cat(vs, 0))
    for j in sorted(gdn_o):
        rs, cs = [], []
        for p in pasts:
            _, g = e5.split_layers(p)
            rs.append(e5._get(g[j].recurrent_states))
            cs.append(e5._get(g[j].conv_states))
        e5._set_like(gdn_o[j], "recurrent_states", torch.cat(rs, 0))
        e5._set_like(gdn_o[j], "conv_states", torch.cat(cs, 0))

    # cac truong int dem theo do dai cache: dat theo T_max
    for k_, v_ in list(vars(out).items()):
        if isinstance(v_, int) and not isinstance(v_, bool) and v_ in lens:
            setattr(out, k_, T_max)
    for l in out.layers:
        for k_, v_ in list(vars(l).items()):
            if isinstance(v_, int) and not isinstance(v_, bool) and v_ in lens:
                setattr(l, k_, T_max)
    del k_any
    return out, lens, T_max


@torch.no_grad()
def greedy_batch(model, past, lens, T_max, warms, n_new, stops, pad_id=0):
    """Sinh greedy cho ca lo. warms: danh sach tensor (1, WARM_P) cua tung hang.

    Tra ve danh sach token id da sinh cho tung hang (da cat o token dung).
    """
    B = len(warms)
    W = warms[0].shape[1]
    dev = warms[0].device
    warm = torch.cat(warms, 0)                       # (B, W) — cung do dai
    # mask: 0 o phan dem trai cua cache, 1 cho phan that + phan sinh moi
    mask = torch.zeros(B, T_max + W, dtype=torch.long, device=dev)
    for i, L in enumerate(lens):
        mask[i, T_max - L:] = 1
    # position_ids cua W token warm: tiep noi do dai THAT cua tung hang,
    # KHONG phai T_max — day la cho de sai nhat khi dem trai.
    pos = torch.stack([torch.arange(L, L + W, device=dev) for L in lens])

    o = model(input_ids=warm, past_key_values=past, attention_mask=mask,
              position_ids=pos, use_cache=True)
    cur = o.past_key_values
    inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
    gen = [[int(x)] for x in inp[:, 0]]
    done = [int(x) in stops for x in inp[:, 0]]
    step = W
    for _ in range(n_new - 1):
        if all(done):
            break
        mask = torch.cat([mask, torch.ones(B, 1, dtype=torch.long, device=dev)], 1)
        pos = torch.tensor([[L + step] for L in lens], device=dev)
        o = model(input_ids=inp, past_key_values=cur, attention_mask=mask,
                  position_ids=pos, use_cache=True)
        cur = o.past_key_values
        inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
        step += 1
        for i, x in enumerate(inp[:, 0]):
            if not done[i]:
                gen[i].append(int(x))
                if int(x) in stops:
                    done[i] = True
    del cur, o
    return gen
