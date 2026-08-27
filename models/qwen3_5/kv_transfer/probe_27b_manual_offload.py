"""probe_27b_manual_offload -- do THAT do tre + bo nho cua duong CPU-offload
THU CONG (e5_train.load_4bit_cpu_offload_io, user 2026-08-27 "dau tu ky
thuat vao duong CPU-offload thu cong") -- ke tiep cua probe_27b_context_
levers.py (duong qua accelerate dispatch da vuong loi "meta tensor").

Muc dich: tra loi CAU HOI CON LAI truoc khi dua vao train that -- lm_head
tren CPU (vocab lon) co du NHANH de goi hang nghin lan (moi buoc train)
khong, hay chi dung duoc cho vai lan goi hiem hoi?

Do ca correctness (cache_respected, khong loi) LAN latency (t=...s) o
T=4096/8192/16384, so truc tiep voi baseline GPU-full (dong A trong
probe_27b_context_levers.py: T=4096 OK 20.19GiB 54.15s).
"""

import importlib.util
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "e5_train", Path(__file__).parent / "e5_train.py")
e5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e5)

# hoc phi 2026-08-27 (probe_27b_context_levers.py, cung bug): transformers
# 5.15 update_recurrent_state lam .copy_() IN-PLACE len tensor state ("static
# address for cudagraphs") -- pha vo autograd khi initial_state la tensor
# mang grad. Rebind khi co grad; giu copy_ cho moi ca khac. PHAI vao truoc
# khi goi load_4bit_cpu_offload_io (patch class-level, ap dung toan cuc).
try:
    from transformers.cache_utils import LinearAttentionLayer
    _orig_urs = LinearAttentionLayer.update_recurrent_state

    def _urs(self, recurrent_states, state_idx=0, **kw):
        cur = (self.recurrent_states.get(state_idx)
               if isinstance(self.recurrent_states, dict) else None)
        if cur is not None and (cur.requires_grad or recurrent_states.requires_grad):
            self.recurrent_states[state_idx] = recurrent_states
            return recurrent_states
        return _orig_urs(self, recurrent_states, state_idx, **kw)

    LinearAttentionLayer.update_recurrent_state = _urs
    print("patched update_recurrent_state (rebind khi co grad)")
except ImportError as e:
    print("PATCH_IMPORT_FAIL", e)


def mem():
    import torch
    return torch.cuda.max_memory_allocated() / 2**30


def probe(m, tok, T, tag, n_repeat=3):
    import torch
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(1000, 50000, (1, T), device="cuda")
    try:
        with torch.no_grad():
            past = m(input_ids=ids[:, :-8], use_cache=True,
                     logits_to_keep=1).past_key_values
        for l in past.layers:
            if hasattr(l, "keys") and l.keys is not None:
                l.keys = l.keys.clone().requires_grad_(True)
                l.values = l.values.clone().requires_grad_(True)
            if hasattr(l, "recurrent_states") and l.recurrent_states is not None:
                rs = l.recurrent_states
                if isinstance(rs, dict):
                    l.recurrent_states = {k: v.clone().requires_grad_(True)
                                          for k, v in rs.items()}
                else:
                    l.recurrent_states = rs.clone().requires_grad_(True)
        warm = ids[:, -8:]
        times = []
        for i in range(n_repeat):
            t0 = time.time()
            out = m(input_ids=warm, past_key_values=past, use_cache=True)
            loss = out.logits.float().pow(2).mean()
            loss.backward()
            torch.cuda.synchronize()
            times.append(time.time() - t0)
            print(f"  {tag} T={T} call{i} t={times[-1]:.2f}s")
            for p in [x for x in m.parameters() if x.grad is not None]:
                p.grad = None
        print(f"{tag} T={T} OK peak={mem():.2f}GiB t_first={times[0]:.2f}s "
              f"t_steady={sum(times[1:]) / max(len(times) - 1, 1):.2f}s "
              f"cache_respected={out.past_key_values is not None}")
        return True
    except Exception as e:
        import traceback
        print(f"{tag} T={T} FAIL {type(e).__name__}: {str(e)[:300]}")
        traceback.print_exc()
        return False
    finally:
        import torch
        torch.cuda.empty_cache()


def main():
    import gc
    import torch
    tok, m = e5.load_4bit_cpu_offload_io("Qwen/Qwen3.5-27B")
    # peak-during-load (bao gom moc full-GPU thoang qua TRUOC khi 2 module
    # bi day xuong CPU) khac voi bo nho ON DINH sau do -- do rieng ca hai,
    # keo tranh nham lan da xay ra o ban truoc (bao "peak" = load peak,
    # khong phai steady-state that su tiet kiem duoc).
    peak_during_load = mem()
    gc.collect()
    torch.cuda.empty_cache()
    steady_after_offload = torch.cuda.memory_allocated() / 2**30
    print(f"LOAD manual-offload peak_during_load={peak_during_load:.2f}GiB "
          f"steady_after_offload={steady_after_offload:.2f}GiB "
          f"embed_dev={m.model.embed_tokens.weight.device} "
          f"lmhead_dev={m.lm_head.weight.device}")
    for T in (4096, 8192, 16384):
        probe(m, tok, T, "E-manual-offload")
    del m
    gc.collect()
    torch.cuda.empty_cache()
    print("PROBE_MANUAL_OFFLOAD_DONE")


if __name__ == "__main__":
    main()
