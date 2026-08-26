"""probe_27b_context_levers -- do THAT 2 don bay mo-rong context cho mapper
4B->27B tren L4 (user 2026-08-27: "ko duoc ket luan som phai lam ky").
Doi chung: ladder goc 27B (e6v3_ce.py --ladder) OOM cung o 8192/16384,
4096 OK peak 21.26GiB. Ket qua (chi tiet STATUS.md muc MAPPER 4B->27B --
thu giai phap mo rong context):
  - gradient checkpointing tren backbone: THUA THAT (peak giong het
    baseline) -- OOM do repeat_kv trong 1 lop attention, khong phai
    tich luy qua nhieu lop -- checkpointing khong dung dung cho.
  - CPU offload embed_tokens+lm_head (llm_int8_enable_fp32_cpu_offload):
    tiet kiem that 4.72GiB nhung loi "meta tensor" -- accelerate dispatch
    hook xung dot voi cach mapper tu sua past_key_values ngoai luong
    forward dang ky. Chua dong -- can viet duong nap thu cong (khong qua
    accelerate hook) de test tiep.

Chay: python probe_27b_context_levers.py  (Colab, sau khi da co HF_TOKEN)
"""
import torch, gc, time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

def mem(): return torch.cuda.max_memory_allocated()/2**30

def load(grad_ckpt=False, offload_io=False):
    torch.cuda.reset_peak_memory_stats()
    dm = "cuda"
    qc_kw = dict(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    if offload_io:
        dm = {"model.embed_tokens":"cpu","lm_head":"cpu","model.layers":0,
              "model.norm":0,"model.rotary_emb":0}
        qc_kw["llm_int8_enable_fp32_cpu_offload"] = True
    m = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-27B", device_map=dm,
        quantization_config=BitsAndBytesConfig(**qc_kw))
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    if grad_ckpt:
        m.gradient_checkpointing_enable()
    print(f"LOAD grad_ckpt={grad_ckpt} offload_io={offload_io} peak={mem():.2f}GiB")
    return m

def probe(m, T, tag):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(1000, 50000, (1, T), device="cuda")
    t0 = time.time()
    try:
        with torch.no_grad():
            past = m(input_ids=ids[:, :-8], use_cache=True, logits_to_keep=1).past_key_values
        for l in past.layers:
            if hasattr(l, "keys") and l.keys is not None:
                l.keys = l.keys.clone().requires_grad_(True)
                l.values = l.values.clone().requires_grad_(True)
            if hasattr(l, "recurrent_states") and l.recurrent_states is not None:
                rs = l.recurrent_states
                if isinstance(rs, dict):
                    l.recurrent_states = {k: v.clone().requires_grad_(True) for k, v in rs.items()}
                else:
                    l.recurrent_states = rs.clone().requires_grad_(True)
        warm = ids[:, -8:].to(m.get_input_embeddings().weight.device
                              if hasattr(m, "get_input_embeddings") else "cuda")
        out = m(input_ids=warm, past_key_values=past, use_cache=True)
        loss = out.logits.float().pow(2).mean()
        loss.backward()
        dt = time.time() - t0
        print(f"{tag} T={T} OK peak={mem():.2f}GiB t={dt:.2f}s "
              f"cache_respected={out.past_key_values is not None}")
        return True
    except Exception as e:
        import traceback
        print(f"{tag} T={T} FAIL {type(e).__name__}: {str(e)[:300]}")
        traceback.print_exc()
        return False
    finally:
        torch.cuda.empty_cache()

m = load(grad_ckpt=False, offload_io=False)
probe(m, 4096, "A-baseline")
probe(m, 8192, "A-baseline")
del m; gc.collect(); torch.cuda.empty_cache()

m = load(grad_ckpt=True, offload_io=False)
probe(m, 4096, "B-gradckpt")
probe(m, 8192, "B-gradckpt")
del m; gc.collect(); torch.cuda.empty_cache()

m = load(grad_ckpt=False, offload_io=True)
probe(m, 4096, "C-offload")
probe(m, 8192, "C-offload")
del m; gc.collect(); torch.cuda.empty_cache()

m = load(grad_ckpt=True, offload_io=True)
probe(m, 8192, "D-both")
probe(m, 16384, "D-both")
del m; gc.collect(); torch.cuda.empty_cache()
print("PROBE27B_DONE")
