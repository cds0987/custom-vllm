# Deep Research: Transfer / Reuse / Transform KV Cache giữa các LLM khác nhau

**Ngày biên soạn:** 2026-08-15
**Phạm vi:** cross-model KV cache transfer, cross-model representation transfer, shared/compatible KV architecture, partial recomputation, và các baseline same-model reuse.
**Nguyên tắc:** mọi claim quan trọng đều gắn nguồn primary (arXiv / official GitHub / official project page). Những chỗ không verify được đều ghi rõ `Not reported` hoặc `I could not verify this claim from primary sources.`

---

## 1. Executive Summary

**Câu trả lời ngắn cho câu hỏi trung tâm:**

State-of-the-art hiện tại (tính đến 2026-08) của việc chuyển KV cache từ LLM A sang LLM B để B decode **mà không recompute prefill** là:

> **Cross-Model KV Cache Transfer (NVIDIA, arXiv:2608.03893, 08/2026)** — một **closed-form per-head ridge regression mapper**, gradient-free, fit từ ~500 sequence calibration, đạt **73–98% retention** accuracy của target trên 4/6 model pair, với **2.7–25× nhanh hơn re-prefill**. Đây là work đầu tiên và duy nhất tôi tìm được đạt **zero-recompute, cross-scale, training-free** đồng thời.

Các điểm chốt:

1. **Chỉ có ĐÚNG MỘT paper** thực sự làm "Model A prefill → mapper → Model B decode, không recompute gì cả, giữa hai model khác kích thước": paper NVIDIA nói trên. Tất cả work khác hoặc (a) yêu cầu cùng architecture (DroidSpeak), (b) vẫn recompute một phần (DroidSpeak, ProxyKV), (c) không skip prefill mà chỉ *làm giàu* KV (C2C, LCF), hoặc (d) chỉ transfer attention pattern chứ không phải KV values (IAM).

2. **Cấu trúc tuyến tính là phát hiện quan trọng nhất.** KV cross-model giữa các model cùng family có structure tuyến tính đáng kể: một source layer giải thích 56% variance của target keys và 32% của values (Qwen3 14B→32B), lên **79% / 65%** khi dùng nhiều source layer. Điều này nghĩa là **không cần train adapter bằng gradient descent** — ridge regression closed-form là đủ cho nhiều cặp.

3. **Nhưng "variance explained" KHÔNG dự đoán được chất lượng.** Paper NVIDIA đo được attention-output cosine tương quan với HellaSwag retention ở Pearson **r = +0.57**, trong khi **R² chỉ đạt r = −0.20** (tức là *âm*). Đây là kết quả phản trực giác quan trọng nhất trong toàn bộ literature: **error placement quan trọng hơn error magnitude**.

4. **Ràng buộc kiến trúc còn rất chặt.** SOTA hiện tại chỉ hoạt động trên "matched-KV pairs" — cùng số KV head và cùng head dimension. Cross-family (Llama→Qwen), khác GQA config, khác tokenizer đều **chưa có solution zero-recompute nào được chứng minh**.

5. **Không có source code cho SOTA.** Paper NVIDIA: `Paper released, but no official implementation was found.` Code chỉ có ở C2C (thu-nics/C2C), IAM (QQQ-yi/IAM), LatentMAS, và DroidSpeak (tích hợp vLLM+LMCache).

6. **Chưa có serving engine nào support cross-model KV transfer ở dạng general.** Duy nhất một dạng hẹp đã vào vLLM: cross-model prefix cache reuse giữa base model và Activated-LoRA adapter (arXiv:2512.17910).

**Xếp hạng nhanh:**

| Tiêu chí | Winner |
|---|---|
| Most practical today | DroidSpeak (same-arch, đã tích hợp vLLM/LMCache) |
| Most promising research direction | Cross-Model KV Transfer (NVIDIA ridge mapper) |
| Best quality | Cross-Model KV Transfer + MLP mapper (>90% HellaSwag retention trên cả 4 pair thử nghiệm) |
| Lowest overhead | Cross-Model KV Transfer (gradient-free, không cần training) |
| Most scalable | LatentAlign (Google DeepMind) — O(N) adapter cho N model, extensible zero-shot |
| Most likely to integrate into vLLM | aLoRA cross-model prefix reuse (đã có PR-level implementation) |

---

## 2. Problem Definition

### 2.1 Tại sao cross-model KV transfer quan trọng

Bài toán được NVIDIA phát biểu chính xác nhất: production LLM serving ngày càng dùng (a) **long agentic sessions** nơi context tích lũy qua nhiều turn, và (b) **multi-model orchestration** cho cost-quality cascading, mid-conversation switching, và routing. Cả hai xu hướng đều làm tăng chi phí **prefill**, và mỗi lần swap model thì receiver phải re-prefill toàn bộ context đã tích lũy. **Prefix caching chỉ giải quyết được trong phạm vi một model.**

Vì output của prefill *chính là* KV cache, nên tái sử dụng nó cross-model quy về một **representation problem**: biến đổi KV cache của model này thành format model kia mong đợi.

Con số minh hoạ mức độ hấp dẫn: trên A100 với Llama-3.1-8B-Instruct, reuse KV cache cho input 40K token giảm prefill latency từ **4s xuống 0.08s** (DroidSpeak §3).

**Use case matrix:**

```
Model cascading      : small LLM xử lý → khó → transfer KV → large LLM
Dynamic routing      : 7B → 14B → 32B → 70B, không prefill lại mỗi tầng
Multi-agent          : agent A nói với agent B, chung conversation history
Personalized serving : nhiều fine-tune của cùng base, chung news/context
Model versioning     : v1 → v2 của cùng model, cache cũ vẫn dùng được
Edge → Cloud         : edge model prefill, cloud model decode
Speculative decoding : draft ↔ target state sharing
Multi-tier serving   : cluster GPU khác nhau chạy model size khác nhau
```

### 2.2 Exact vs Approximate KV reuse

| Loại | Điều kiện | Quality | Ví dụ |
|---|---|---|---|
| **Exact** | Cùng model, cùng prefix, cùng position | Lossless | PagedAttention prefix caching, RadixAttention |
| **Approximate (positional)** | Cùng model, prefix khác position | Cần position fix | Prompt Cache, TurboRAG, KVLink |
| **Approximate (cross-attention loss)** | Cùng model, chunk độc lập | Cần selective recompute | CacheBlend, KVCOMM |
| **Approximate (cross-model, same arch)** | Khác weights, cùng shape | Cần recompute ~11% layer | DroidSpeak |
| **Approximate (cross-model, cross-scale)** | Khác L, khác d_model | Cần learned/fitted mapping | **Cross-Model KV Transfer, C2C, LatentAlign** |
| **Không khả thi hiện nay** | Khác tokenizer / khác attention family | — | — |

Điểm cần nhấn: **direct reuse không qua transform là thảm hoạ**. DroidSpeak Insight 1: "Reusing the whole KV cache between models leads to a huge loss in accuracy" — HotpotQA mất >50% accuracy points trên hầu hết model pair, **ngay cả khi hai model cùng architecture và chỉ khác nhau ở fine-tuning**.

---

## 3. Taxonomy of Techniques

### Category A — True Cross-Model KV Transfer (zero recompute)

`KV_A ──W / adapter──► KV_B`, target **không chạy layer nào** trên prompt.

| Work | Org | Zero-recompute | Cross-scale | Training-free |
|---|---|---|---|---|
| **Cross-Model KV Cache Transfer** (2608.03893) | NVIDIA | ✅ | ✅ | ✅ (closed-form ridge) |
| **LatentAlign / Latent Space Comm. via K-V Cache Alignment** (2601.06123) | Google DeepMind | ✅ (prefix cache) | ✅ | ❌ (adapter training) |
| **Latent Cache Flow (LCF)** (2605.22863) | Columbia | ⚠️ (một phần; LCF-X) | ✅ | ❌ |

### Category B — Cross-Model Representation Transfer (hidden state / activation / attention)

| Work | Transfer object | Ghi chú |
|---|---|---|
| **C2C / Cache-to-Cache** (2510.03215) | KV cache (fusion, không thay thế) | Receiver **vẫn prefill**; đây là enrichment |
| **IAM** (2507.11953) | Attention **maps**, không phải KV values | prefill +15%, KV −22.1% |
| **LatentMAS** | KV-cache + hidden states | zero-training latent handoff |
| **DFlash / EAGLE-3 / Medusa** | Target hidden states → drafter | Speculative decoding |
| **GliDe / LongSpec** | Target KV → draft cross-attention | Draft model đọc target KV trực tiếp |

### Category C — Shared / Compatible KV Architecture

- **LatentAlign**: học một *global shared latent space* Σ, mỗi model có adapter in/out. Số parameter chỉ tăng **tuyến tính** theo số model.
- **KV Cache Transform Coding** (2511.01815, ICLR 2026): chứng minh key head trong cùng model nằm chung subspace up to orthogonal transform (Procrustes).
- Joint-training / distillation để KV compatible: **chưa tìm thấy paper nào làm điều này ở scale LLM.** → research gap lớn.

### Category D — Partial Recomputation

- **DroidSpeak**: reuse phần lớn layer, recompute contiguous group của "critical layers" (~11% layer là critical). Cần transfer thêm **E cache** (embedding) tại transition layer — E cache lớn gấp 2× KV cache với Mistral-7B/Llama-3-8B family và gấp 4× với Llama-3.1-70B do GQA.
- **ProxyKV** (2605.16360): proxy model nhỏ hướng dẫn pruning KV cho target lớn; HybridAxialMapper chỉ chiếm 4.7–6.8% prefill wall time; 3.21× prefill speedup.
- **CacheBlend**: recompute selective **token** (không phải layer) — same-model.

### Category E — Same-Model KV Reuse (BASELINE ONLY — không phải cross-model)

PagedAttention/vLLM, RadixAttention/SGLang, Prompt Cache, CacheGen, TurboRAG, KVLink, CacheBlend, KVCOMM, CacheTune, SparseX. **Không được nhầm nhóm này với cross-model transfer.**

### Category F — KV Compression (chỉ liên quan gián tiếp)

xKV (cross-layer SVD), KIVI, SnapKV, PyramidKV, H2O, StreamingLLM, KV Transform Coding, NVFP4 KV cache. Liên quan ở chỗ: **low-rank / shared-basis structure của KV là cùng một hiện tượng toán học** làm cho cross-model linear mapping khả thi.

---

## 4. Paper Landscape

### 4.1 ⭐ Cross-Model KV Cache Transfer in LLM Families (NVIDIA)

| Field | Information |
|---|---|
| Paper | Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse |
| Authors | Taekyung Heo, Rasoul Shafipour, Ritchie Zhao, Maximilian Golub, Mohammad Mahdi Kamani, Ritika Borkar, Makesh Tarun Chandran, Pantea Zardoshti, Bita Darvish Rouhani |
| Organization | **NVIDIA** |
| Year | 2026 (submitted 4 Aug 2026) |
| Venue | arXiv preprint (cs.LG). Venue chính thức: Not reported |
| arXiv | arXiv:2608.03893v1 |
| Paper URL | https://arxiv.org/abs/2608.03893 |
| Project page | Not found |
| GitHub | **Not found** |
| Code available? | **No** — *Paper released, but no official implementation was found.* |
| Model family | Qwen3, Llama 3.1, Ministral 3 |
| Source model | Qwen3-8B/14B; Llama-3.1-8B; Ministral-3B/8B |
| Target model | Qwen3-32B; Llama-3.1-70B; Ministral-8B/14B |
| Same architecture? | Cùng family, khác scale (không cần identical) |
| Different model size? | ✅ (8B→70B là tỉ lệ param lớn nhất được test) |
| Different hidden dimension? | ✅ |
| Different number of layers? | ✅ (L_s ≠ L_t) |
| Different number of heads? | Q heads khác; **KV heads phải bằng nhau** (matched-KV) |
| GQA/MQA/MHA compat | Chỉ GQA matched: n_kv^s = n_kv^t **và** d_h^s = d_h^t |
| Transfer object | **KV (cả K và V)** |
| Transformation | Per-(target layer, head) ridge regression, multi-source-layer concat, RoPE-stripped content space |
| Training required? | **Không** — closed-form solve, gradient-free |
| Extra parameters | **1.01–3.36 B params**, storage **4–12 GB** per pair |
| Prefill recomputation required? | **Không (zero)** |
| Quality loss | Tier 1: retention 73–98% (4 pair). Tier 2: 42–44% (2 pair Ministral), 11–15% khi floor-normalized |
| TTFT improvement | Mapper chạy **2.7–25× nhanh hơn re-prefill** (small→large direction) |
| Throughput improvement | Not reported |
| Memory overhead | 4–12 GB mapper weights per model pair |
| Hardware | 8×H100 node (fit mapper: 47–87 phút/pair) |
| Main limitation | Chỉ matched-KV pairs; mapper 4–12GB/pair; 2/6 pair thất bại nặng với ridge; chưa có code |

### 4.2 Cache-to-Cache (C2C)

| Field | Information |
|---|---|
| Paper | Cache-to-Cache: Direct Semantic Communication Between Large Language Models |
| Authors | Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, Yu Wang |
| Organization | Tsinghua University (thu-nics), Infinigence AI, CUHK, Shanghai AI Lab, SJTU |
| Year | 2025 (arXiv Oct 2025) |
| Venue | **ICLR 2026** |
| arXiv | arXiv:2510.03215 |
| Paper URL | https://arxiv.org/abs/2510.03215 |
| Project page | https://fuvty.github.io/C2C_Project_Page/ |
| GitHub | https://github.com/thu-nics/C2C |
| Code available? | **Yes** (+ fuser weights trên HuggingFace collection, live gradio demo) |
| Model family | Qwen2.5, Qwen3, Llama3.2, Gemma3 |
| Same architecture? | Không cần — có layer mapping + token alignment |
| Transfer object | **KV cache (fusion vào receiver KV, không thay thế)** |
| Transformation | Cache Fuser = Projection + Dynamic (input-aware head) Weighting + Learnable per-layer Gate, residual |
| Training required? | **Có** — freeze cả hai model, train riêng C2C module bằng next-token-prediction loss |
| Prefill recompute required? | **CÓ** — receiver vẫn prefill. **Đây KHÔNG phải prefill-skipping.** |
| Quality | +6.4–14.2% avg accuracy vs individual model; +3.1–5.4% vs text-to-text |
| Speedup | ~2.0× latency vs text-to-text communication (không phải vs no-communication) |
| Main limitation | Cần cả hai model prefill; adapter lớn; token-aligned context bắt buộc |

**Oracle experiment đáng chú ý:** C2C train một 3-layer MLP map KV-Cache từ Qwen3-4B → Qwen3-0.6B. t-SNE cho thấy raw KV của hai model **cách xa nhau** trong representation space, nhưng sau transform thì mapped KV **nằm bên trong** target space. Tuy nhiên — và đây là quan sát về information bottleneck — **transformed cache chỉ chiếm một subset nhỏ hơn** của target space. C2C kết luận: "source model's semantic information cannot fully cover the target's, despite the source being larger."

### 4.3 LatentAlign — Latent Space Communication via K-V Cache Alignment

| Field | Information |
|---|---|
| Authors | Lucio M. Dery, Zohar Yahav, Henry Prior, Qixuan Feng, Jiajun Shen, Arthur Szlam |
| Organization | **Google DeepMind** |
| Year | 2026 (4 Jan 2026) |
| arXiv | arXiv:2601.06123 |
| GitHub | Not found — *Paper released, but no official implementation was found.* |
| Model family | Gemma-2, self-pretrained 100M/200M/400M (không phải Gemma-2 public checkpoints) |
| Different layers? | ✅ (4 layers vs 16 layers) |
| Transfer object | **K-V cache blocks** |
| Transformation | Hai adapter mỗi model: T[m→Σ] và T[Σ→m], qua **shared implicit latent space Σ** |
| Architecture của adapter | Multi-layer **cross-attention** transformer (Gemma-2 arch modified), 32 heads, head_dim 64, ~¼ kích thước base model |
| Training | 50k steps, AdamW, lr sweep {1e-3,1e-4,1e-5}, batch 256–1024, mC4 |
| Loss | Reconstruction loss + **Suffix LM loss**; kết luận: **reconstruction loss là không cần thiết** khi đã có suffix LM loss |
| Prefill recompute? | Không (prefix cache được translate rồi decode tiếp) |
| Quality | Vượt base model performance trong nhiều setting; **cyclic translation qua Σ đôi khi còn TỐT HƠN cache gốc** |
| Extensibility | ✅ Thêm model thứ 4 chỉ cần học 2 adapter, **zero-shot** translate được path chưa từng train |
| Main limitation | Model rất nhỏ (100M–400M), chưa scale lên LLM production; adapter tốn ~¼ model size mỗi model |

**Kết quả ablation quan trọng (Table 1 của paper):** identity mapping cho kết quả tệ (4.941 vs base 3.089) *ngay cả khi hai model chỉ khác random seed*. **Linear map** đạt 3.090/3.110 — tức là **ngang hoặc vượt base model**, khớp với literature về latent space geometry equivalence modulo linear transformation. Caveat: linear map cần up-project 8×/24×.

### 4.4 DroidSpeak

| Field | Information |
|---|---|
| Authors | Yuhan Liu, Yuyang Huang, Jiayi Yao, Shaoting Feng, Zhuohan Gu, Kuntai Du, Hanchen Li, Yihua Cheng, Junchen Jiang (UChicago); Shan Lu, Madan Musuvathi, Esha Choukse (Microsoft) |
| Year | v4: 14 Jul 2025 |
| arXiv | arXiv:2411.02820 |
| Same architecture? | **Bắt buộc identical architecture** |
| Different model size? | ❌ |
| Transfer object | KV cache + **E cache** (embedding tại transition layer) |
| Transformation | **Không có transform** — reuse trực tiếp + selective layer recomputation |
| Training required? | Không, nhưng cần **offline profiling** O(L²); 3 giờ trên A100 cho model 32 layer |
| Prefill recompute? | **Có** — recompute contiguous group of critical layers (~11% layers là critical) |
| Quality | Negligible loss trên F1 / Rouge-L / code similarity |
| TTFT / prefill | **1.7–3.1×** giảm prefill latency (avg 2.1×); 2.7× TTFT trong agentic coding workflow |
| Throughput | **up to 4×** |
| Hardware | 2× Azure Standard_ND96amsr_A100_v4 (8×80GB A100 mỗi node), InfiniBand |
| Implementation | ~3K LoC Python, PyTorch 2.0, CUDA 12.0, **LMCache 0.1.4 + vLLM**, deploy trên vLLM Production Stack + Kubernetes |
| Main limitation | Không support cross-foundation-model (KV khác size); data drift trong profiling config |

**Ba insight nền tảng của DroidSpeak (giá trị vượt xa paper này):**
1. Reuse toàn bộ KV giữa hai model → mất accuracy khổng lồ.
2. **Chỉ một tập nhỏ layer nhạy cảm** với KV cache deviation (~11%), và identity của chúng ổn định theo model pair.
3. Variance của sensitivity giữa các input **chỉ đáng kể ở critical layers**.

**Bài học kiến trúc quan trọng:** recompute các critical layer *không liền kề* là sai lầm — mỗi transition point (reuse→recompute) đòi hỏi load E cache của sender, và error từ E cache đó propagate xuống tất cả layer sau. Nhiều transition point ⇒ nhiều nguồn deviation cộng dồn. Vì thế DroidSpeak recompute **contiguous group**.

### 4.5 IAM — Attention Mapping between Different-scale LLMs

| Field | Information |
|---|---|
| Authors | Yi Zhao, Zuchao Li, Hai Zhao |
| Venue | **ACL 2025** |
| arXiv | arXiv:2507.11953 |
| GitHub | https://github.com/QQQ-yi/IAM — **Code available: Yes** |
| Transfer object | **Attention matrices (maps)**, KHÔNG phải KV values |
| Quality | "without appreciably sacrificing performance" |
| Prefill | +15% acceleration |
| Memory | −22.1% KV cache usage |
| Main limitation | Không skip prefill hoàn toàn; transfer attention pattern nên vẫn cần V của target |

### 4.6 Latent Cache Flow (LCF)

| Field | Information |
|---|---|
| Authors | Maximillian Rossi et al. (equal contribution) |
| Organization | Columbia University |
| arXiv | arXiv:2605.22863 |
| Patent | **Patent pending — U.S. Provisional Patent Application No. 64/065,974** |
| Transformation | Single **shared key-value pipeline** (thay vì dual pipeline như C2C) + **low-dimensional latent bottleneck** |
| Extra parameters | **~4% kích thước adapter của C2C** |
| Đóng góp thêm | **LCF-X**: pooling module cho phép communication khi sharer/receiver có context **khác nhau** (C2C yêu cầu context identical) |
| GitHub | Not found |

### 4.7 Efficient Multi-Adapter LLM Serving via Cross-Model KV-Cache Reuse with Activated LoRA

| Field | Information |
|---|---|
| Authors | Allison Li (MIT), Kristjan Greenewald, Thomas Parnell (IBM Research), Navid Azizan (MIT) |
| arXiv | arXiv:2512.17910 |
| Transfer object | Prefix KV cache giữa **base model ↔ aLoRA adapted model** |
| Transformation | **Không có transform** — aLoRA thiết kế sao cho cache tương thích sẵn (Category C) |
| Implementation | **Mở rộng vLLM**: base-aligned block hashing + activation-aware masking trong model execution path |
| Quality | Lossless theo thiết kế |
| Latency | **up to 58× end-to-end latency reduction; >100× TTFT** vs standard LoRA baseline |
| Status | Integrated into production-grade inference stack |
| Main limitation | Chỉ áp dụng cho aLoRA (adapter của cùng base model), không phải hai model độc lập |

### 4.8 Các work liên quan khác (bảng gọn)

| Work | arXiv | Category | Code | Điểm chính |
|---|---|---|---|---|
| ProxyKV | 2605.16360 | D | Not found | HybridAxialMapper; 3.21× prefill speedup; mapper = 4.7–6.8% prefill time |
| KV Cache Transform Coding | 2511.01815 (ICLR'26) | F/C | Not found | Procrustes alignment giữa head; PCA basis; 200K token calibration |
| xKV | 2503.18893 (ICML'26) | F | https://github.com/abdelfattah-lab/xKV | Cross-layer SVD; 8× compression; 4.23× e2e speedup |
| LatentMAS | — | B | https://github.com/Gen-Verse/LatentMAS | ICML 2026 Spotlight; zero-training latent handoff; patch vLLM internals |
| GliDe | — | B | Not verified | Draft model cross-attend vào target KV cache |
| LongSpec | 2502.17421 | B | Not verified | Kế thừa GliDe; share embedding + LM head |
| When Hidden States Drift | 2604.26412 | B | Not found | KV-Reuse Hypothesis; Qwen3-8B; **end-to-end speedup vẫn marginal** |
| Cross-layer Attention Sharing (LISA) | 2408.01890 | C/F | Not verified | FFN nhỏ align head giữa layer kề + low-rank difference |
| SwiftCache | 2606.16135 | D/E | Not found | Heterogeneous KV sharing multi-turn |
| Prefill-as-a-Service | 2604.15039 | Systems | Not found | KV throughput 60 Gbps ở 32K với dense GQA |
| HBM Is Not All You Need | 2606.29986 | Systems | Not found | Số liệu chi phí transfer KV qua RDMA |

---

## 5. Deep Dive Into Cross-KV (NVIDIA, arXiv:2608.03893)

### 5.1 Architecture

```
Input tokens x = (x_1 ... x_T)
        │
        ▼
   Model S (source)  ──► C_S = {K_s^{l,h}, V_s^{l,h}}
        │
        ▼
   [strip source RoPE from K]        R_Θs^{-1}(t)
        │
        ▼
   [per target layer l: chọn top-k source layer, concat]
        │
        ▼
   [per (l,h): ridge W_K^{l,h}, W_V^{l,h}]
        │
        ▼
   [re-apply target RoPE to K]       R_Θt(t)
        │
        ▼
   Ĉ_T  ──►  Model T decode ngay, KHÔNG prefill
```

Mục tiêu formal:

```
m( T(x ; Ĉ_T) )  ≈  m( T(x ; C_T) )
```

với `m` là metric của downstream task τ. **Điểm quan trọng về phương pháp luận:** paper đo transfer quality bằng **downstream accuracy của target model**, không phải bằng reconstruction metric. Đây chính là lý do họ phát hiện được R² không đáng tin.

### 5.2 Mathematical formulation

**Probe tuyến tính (§2.3), một source layer:**

```
Ĉ_t^{l,h} = C_s^{l',h} · W + b
W ∈ R^{d_h^s × d_h^t},   b ∈ R^{d_h^t}
```

fit ở **token level** — mỗi calibration token là một observation.

**Production mapper (§3.1), multi-source-layer:**

```
K̂_t^{l,h} = X_K^l · W_K^{l,h} + b_K^{l,h}
V̂_t^{l,h} = X_V^l · W_V^{l,h} + b_V^{l,h}

W_K^{l,h} ∈ R^{(k · n_kv^s · d_h^s) × d_h^t}
```

với

```
X_K^l = [ K̄_s^{l_1} ‖ K̄_s^{l_2} ‖ ... ‖ K̄_s^{l_k} ]
K̄_s^{l_i} ∈ R^{T × (n_kv^s · d_h^s)}
```

Chú ý: `K̄_s^{l_i}` là concat của **TẤT CẢ key head** của source layer l_i. Nghĩa là mỗi target head đọc từ **toàn bộ** head của source — đây là cơ chế **cross-head information flow** ngầm, không cần head alignment tường minh.

### 5.3 Layer mapping

**Không dùng heuristic (l/L_s ≈ l'/L_t).** Với mỗi target layer l, chọn **top-k source layer** theo head-averaged R², trung bình trên RoPE-stripped keys và values.

Bằng chứng cần nhiều source layer:
- Trên Qwen3 14B→32B, `k=1` chỉ bắt được **66%** của R² tại `k=all` cho K_stripped, và **42%** cho V.
- Gain lớn nhất từ k=1→k=4; R² gần bão hoà ở k=6.
- k được sweep trên {1,2,4,6,8,10,12,16,20,24,all}, chọn per-pair.
- k đã chọn: Qwen3 14B→32B: **8**; Qwen3 8B→32B: **12**; Llama3.1 8B→70B: **20**; Ministral 3B→8B: **all**; Ministral 3B→14B: **20**; Ministral 8B→14B: **12**.

**Đây là contributor lớn nhất trong ba component.** Ablation: giảm k từ 8 → 1 làm K R² rơi từ **0.79 → 0.56**, và accuracy sụp đổ (ARC-C 61.60 → 27.65, GSM8K 90.98 → 0.38, PPL 7.33 → 22.73).

**Hệ quả nhận thức:** thông tin mà target layer l cần **không nằm ở một source layer tương ứng** — nó phân tán qua nhiều source layer. Đây là bằng chứng trực tiếp chống lại giả thuyết "layer-to-layer correspondence" đơn giản.

### 5.4 Head mapping

Không có head-to-head alignment tường minh. Cơ chế thực tế:
- Mapper độc lập cho **mỗi (target layer, target head)** → `L_t × n_kv^t` bài toán ridge.
- Input là **all source heads** của k layer đã chọn ⇒ mỗi target head là **many-to-one** combination của toàn bộ source head.
- Tất cả head trong cùng target layer **dùng chung tập source layer đã chọn**, "enabling cross-head information flow".
- **Không share parameter giữa head, cũng không share giữa K và V.**

**Ràng buộc:** phương pháp chỉ được validate trên "matched-KV" (n_kv^s = n_kv^t và d_h^s = d_h^t). Về mặt toán học, W ∈ R^{(k·n_kv^s·d_h^s) × d_h^t} **không đòi hỏi** d_h^s = d_h^t, nhưng paper không report kết quả cho trường hợp mismatched.

### 5.5 W matrices — kiểm kê chính xác

Câu trả lời cho từng câu hỏi ở §7 của brief:

| Câu hỏi | Trả lời |
|---|---|
| 1 W cho toàn model? | ❌ |
| 1 W cho mỗi layer? | ❌ |
| 1 W cho mỗi KV head? | ✅ — **W riêng cho mỗi (target layer, target head)** |
| W riêng cho K và V? | ✅ — **W_K và W_V hoàn toàn tách biệt** |
| W shared giữa layer? | ❌ |
| W shared giữa K và V? | ❌ ("No parameters are shared across heads or between K and V") |

**Tổng số parameter:**

```
N_params = 2 · L_t · n_kv^t · (k · n_kv^s · d_h^s) · d_h^t
```

(hệ số 2 = K + V). Paper report thực tế: **1.01–3.36 B parameters**, storage **4–12 GB** mỗi model pair.

Ví dụ tính tay cho Qwen3 14B→32B, k=8, giả sử n_kv = 8, d_h = 128, L_t = 64:
```
2 × 64 × 8 × (8 × 8 × 128) × 128 = 2 × 64 × 8 × 8192 × 128 ≈ 1.07 B params
```
→ khớp với khoảng 1.01–3.36B mà paper report.

**Đây là một overhead nghiêm túc:** một mapper 4–12GB cho **mỗi cặp model có hướng**. Với hệ thống routing 4 tier (7B/14B/32B/70B) và transfer hai chiều, số cặp là 12 → có thể lên tới ~144GB mapper weights.

### 5.6 Training / Fitting

**Objective:** ridge regression (Tikhonov), MSE.

```
W* = (Xᵀ X + λ I)^{-1} Xᵀ Y ,   λ = 0.01
```

- X ∈ R^{N × (k·n_kv^s·d_h^s)}, Y ∈ R^{N × d_h^t}
- X, Y được **center** trước khi solve, bias phục hồi bằng `b = Ȳ − X̄ W*`
- Lý do dùng ridge thay OLS: feature dimension có thể lên hàng chục nghìn ở k lớn, và các top-k source layer **theo định nghĩa là correlated**, nên XᵀX gần singular.

**Calibration dataset:**
```
prompt (FineWeb-Edu)
 ├─► Model S ─► KV_S
 └─► Model T ─► KV_T   (đã strip target RoPE cho K)
         │
         ▼
   ridge fit W per (l,h)
```
- **500 sequence FineWeb-Edu, mỗi sequence 1024 token**
- stride-4 subsample → **N ≈ 128K token** mỗi target head
- Fit time: **47–87 phút/pair** trên một node 8×H100, **không có gradient training**
- Bottleneck: forming XᵀX là O(N·d_s²) với d_s = k·n_kv^s·d_h^s, và **được tính một lần cho mỗi target layer, dùng chung cho các head** ⇒ scaling sub-linear theo số head

**Robustness của calibration:**
- Sweep λ 4 bậc độ lớn: vùng phẳng rộng, chỉ sụp đổ ở λ=1
- Sweep N từ 50→1000 sequence: phẳng sau N=200; **N=50 vẫn trong ~1.6pp của production**
- **Domain là trục duy nhất có cost thật:** calibrate trên CodeAlpaca làm HellaSwag rơi 5.24pp, còn Wikipedia thì trong noise

**Nonlinear variant (MLP):** per-(target layer, head, K|V), 2 hidden layer 1024 units ReLU, Adam, **cùng MSE loss với ridge**, drop-in replacement lúc inference.

### 5.7 RoPE handling — chi tiết quan trọng nhất về mặt kỹ thuật

KV cache lưu key **đã xoay**: `k_RoPE(t) = R_Θ(t) · k_content`. Nếu fit mapper trên key đã xoay thì W bị **gắn chặt vào phân bố position của lúc fit** (1024 token).

Pipeline content-space:

```
K̂_t = ( K_s · R_Θs^{-1}(t) · W_K + b_K ) · R_Θt(t)
```

Trong calibration, Y được tạo bằng cách **strip target RoPE khỏi ground-truth key của target**, nên W_K được fit hoàn toàn trong không gian position-free. Vì R_Θ trực giao nên nghịch đảo là **exact và gần như miễn phí**.

**Ablation cực kỳ đáng chú ý (Table 2, Qwen3 14B→32B):**

| Configuration | ARC-C | HellaSwag | WinoGrande | MMLU | GSM8K | PPL |
|---|---|---|---|---|---|---|
| Full (k=8, ridge, content-space) | 61.60 | 80.70 | 68.98 | 78.09 | 90.98 | 7.33 |
| − inference RoPE (mismatch) | 44.97 | 75.39 | 56.59 | **25.79** | **4.17** | 7.70 |
| − all RoPE (coupled fit + inference) | 61.09 | 80.73 | 68.59 | 77.70 | 90.98 | 7.35 |
| − RoPE − cross-layer (k=1) | 27.65 | 44.81 | 51.78 | 26.07 | 0.38 | 22.73 |
| − RoPE − cross-layer − ridge | 36.43 | 62.26 | 51.22 | 51.26 | 1.44 | 9.86 |

Đọc bảng này cẩn thận:
- **Fully-coupled RoPE (fit + inference) hoạt động tốt ngang content-space** trên short-context benchmark (61.09 vs 61.60). Lợi ích của content-space **không phải** ở accuracy ngắn hạn mà ở **generalization sang context dài hơn** (paper serve prompt tới 32k).
- **Mismatch giữa fit và inference thì thảm hoạ**: MMLU về random (25.79), GSM8K về 4.17.
- **Content-space mapping là benchmark-specific**: MMLU và GSM8K sụp về random, còn HellaSwag chỉ rơi ~5pp. → tasks đòi reasoning nhiều bước nhạy cảm với position hơn hẳn.

### 5.8 Kết quả chính

| Family | Pair (k) | Avg retention | Avg_fn (floor-normalized) | ARC-C | HellaSwag | WinoGrande | MMLU | GSM8K |
|---|---|---|---|---|---|---|---|---|
| *Chance floor* | | — | — | 25% | 25% | 50% | 25% | ≈0% |
| Qwen3 | 14B→32B (8) | **97.6%** | 96.3% | 101.0% | 97.6% | 98.5% | 95.0% | 95.6% |
| Qwen3 | 8B→32B (12) | 87.5% | 80.7% | 94.0% | 95.2% | 91.0% | 88.5% | 68.8% |
| Llama 3.1 | 8B→70B (20) | 72.8% | 62.9% | 90.9% | 94.4% | 87.1% | 73.3% | 18.2% |
| Ministral 3 | 3B→8B (all) | 76.2% | 65.9% | 90.6% | 93.3% | 91.3% | 69.4% | 36.6% |
| Ministral 3 | 3B→14B (20) | **44.2%** | **14.7%** | 43.6% | 68.0% | 74.0% | 32.0% | 3.2% |
| Ministral 3 | 8B→14B (12) | **41.6%** | **11.1%** | 40.7% | 58.7% | 74.2% | 32.7% | 1.6% |

**Ba quan sát nghiêm túc từ bảng này:**

1. **GSM8K là canary.** Trong khi HellaSwag retention giữ 93–98% ở hầu hết pair, GSM8K rơi xuống 18.2% (Llama) và 1.6–3.2% (Ministral fail). Multi-step reasoning là thứ chết trước tiên. Nếu bạn chỉ đo HellaSwag, bạn sẽ kết luận sai rằng technique này đã sẵn sàng.

2. **Floor-normalization phơi bày mức độ tệ.** Ministral 8B→14B "41.6%" nghe như còn ~40% năng lực, nhưng floor-normalized chỉ **11.1%** — tức là gần như random trên các multiple-choice benchmark.

3. **Matched-KV là điều kiện cần chứ không đủ.** Cả 6 pair đều matched-KV, nhưng 2 pair vẫn sụp. Paper thừa nhận điều này thẳng thắn: "matched KV correlates with success but does not guarantee it."

**MLP rescue (Table 3):** trên các pair mà ridge thất bại, MLP phục hồi HellaSwag retention **+24.3 đến +36.8 pp**, đẩy cả 4 pair test lên trên **90%**. Nhưng trên pair mà ridge đã thành công, **MLP hơi kém hơn ridge** (Qwen3 14B→32B: 97.6% ridge vs 97.3% MLP).

→ Kết luận: **ridge đủ khi quan hệ cross-model KV vốn đã tuyến tính; MLP chỉ giúp ở nơi ridge thiếu, chứ không dominate.**

---

## 6. Source Code Analysis

Đây là phần tôi phải báo cáo trung thực về giới hạn.

### 6.1 Cross-Model KV Cache Transfer (NVIDIA)

```
Repository:        KHÔNG TÌM THẤY
Commit:            N/A
Relevant folders:  N/A
```

> **Paper released, but no official implementation was found.** Tôi đã tìm trên arXiv abstract page (không có link Code/Data), qua search GitHub, và qua CatalyzeX/PapersWithCode entries. `I could not verify any code release for arXiv:2608.03893.`

Nếu cần reimplement, pipeline từ paper là đủ chi tiết:
```
1. capture_kv.py     : chạy S và T trên cùng 500 seq FineWeb-Edu, dump K,V per (layer,head)
2. strip_rope.py     : apply R_Θ^{-1}(t) lên K của cả S và T
3. probe_r2.py       : OLS single-source (l',l) heatmap → head-averaged R²
4. select_layers.py  : top-k source layer per target layer theo R²
5. fit_ridge.py      : XᵀX accumulate per target layer (shared across heads),
                       solve (XᵀX + λI)^{-1} XᵀY per head
6. runtime_map.py    : strip source RoPE → concat top-k → W_K/W_V → re-apply target RoPE
                       → ghi vào target KV cache layout
```

### 6.2 C2C — https://github.com/thu-nics/C2C

```
Repository:  github.com/thu-nics/C2C   ("Rosetta")
Status:      [OPEN-SOURCE IMPLEMENTATION]
License:     xem repo
Commit:      Not pinned in this report
```
Các artefact được xác nhận từ repo README:
- `script/playground/gradio_demo.py` — live demo side-by-side model comparison
- `live_chat_example.py` — multi-sharer usage (feature từ 12/2025, README ghi rõ "preliminary stages")
- Fuser weights công bố trên **HuggingFace collection**, có script inference tối thiểu để load
- Danh sách "available fuser pairs" cho các cặp Qwen2.5 / Qwen3 / Llama3.2 / Gemma3

> Tôi **không** liệt kê tên class/function cụ thể bên trong repo vì không fetch được cây file để verify. `I could not verify the internal file/class structure of thu-nics/C2C from primary sources.` — cần `git clone` để xác nhận.

### 6.3 IAM — https://github.com/QQQ-yi/IAM

```
Status: [OPEN-SOURCE IMPLEMENTATION]
```
Paper ghi rõ "Our code is available at https://github.com/QQQ-yi/IAM". Cấu trúc file: `Not reported` trong nguồn tôi đọc được.

### 6.4 DroidSpeak

```
Status: [INTEGRATED INTO SERVING ENGINE]  (vLLM + LMCache)
Repository: link bị anonymize trong bản arXiv (double-blind); URL công khai: Not found
```
Nhưng interface được mô tả tường minh trong paper — đây là **contract API thực dụng nhất** cho bất kỳ ai muốn implement cross-model KV trong vLLM:

```python
store(Cache, context, LLM)
    # split KV/E cache theo layer, lưu vào key-value store trên GPU memory
    # key = hash(context text)

fetch(context, LLM, layer_id) -> Cache
    # load KV hoặc E cache của layer tương ứng
    # implemented với torch.distributed để fetch từ remote GPU node

partial_prefill(recompute_config, context) -> text
    # recompute_config = danh sách layer cần recompute
    # gọi fetch_kv cho layer reuse, fetch_e tại transition layer
```

Chi tiết implementation đáng chú ý: **mọi transmission được đặt trên CUDA Stream khác với default compute stream** của PyTorch, cho phép overlap transfer KV với recomputation. Đây chính là cơ chế pipelining giảm TTFT từ 30 → 17 (đơn vị thời gian mô hình hoá trong Figure 13).

### 6.5 LatentMAS — https://github.com/Gen-Verse/LatentMAS

```
Status: [OPEN-SOURCE IMPLEMENTATION], ICML 2026 Spotlight
```
README ghi một cảnh báo rất đáng lưu ý cho bất kỳ ai làm cross-model KV trên vLLM:

> "vLLM does not officially support modifying KV-cache or prompting via latent embeddings. We modify the partial inner package inside vLLM backend for our method implementation."

→ **Đây là bằng chứng trực tiếp cho research gap #18: vLLM hiện chưa có public API để ghi KV cache từ bên ngoài.**

### 6.6 Trạng thái implementation tổng hợp

| Work | Status |
|---|---|
| Cross-Model KV Transfer (NVIDIA) | `[RESEARCH PROTOTYPE]` (theo mô tả trong paper) — nhưng **không có code công khai** |
| C2C | `[OPEN-SOURCE IMPLEMENTATION]` |
| IAM | `[OPEN-SOURCE IMPLEMENTATION]` |
| LatentMAS | `[OPEN-SOURCE IMPLEMENTATION]` (patch vLLM internals) |
| DroidSpeak | `[INTEGRATED INTO SERVING ENGINE]` (vLLM + LMCache, "tested in enterprise settings") |
| aLoRA cross-model reuse | `[INTEGRATED INTO SERVING ENGINE]` — "Integrated into a production-grade inference stack" |
| LatentAlign | `[RESEARCH PROTOTYPE]` — no code found |
| LCF | `[RESEARCH PROTOTYPE]` — no code found, patent pending |

> **Về claim "NVIDIA production đang dùng Cross-KV":**
> **I could not verify this claim from primary sources.** Paper là arXiv preprint từ NVIDIA researchers; không có tuyên bố nào về deployment production, không có TensorRT-LLM/Dynamo integration nào được nhắc tới trong paper hay trong tài liệu NVIDIA công khai mà tôi tìm được.

---

## 7. Systems / Serving Analysis

### 7.1 Cost model

**Không transfer (baseline cascade):**
```
T_no_transfer = T_S,prefill + T_S,decode + T_T,prefill + T_T,decode
```

**Có transfer:**
```
T_transfer = T_S,prefill + T_S,decode + T_move + T_map + T_T,decode
```

**Điều kiện có lợi:**
```
T_move + T_map  <  T_T,prefill
```

Bằng chứng thực nghiệm: mapper của NVIDIA chạy **2.7–25× nhanh hơn re-prefill** ở hướng small→large. Biên độ rộng (2.7 vs 25) phản ánh tỉ lệ kích thước model — 8B→70B thì re-prefill đắt hơn nhiều nên tỉ lệ tiết kiệm cao hơn.

### 7.2 Phân tích theo context length

Prefill FLOPs ~ O(T · P) cho phần dense (P = params) cộng O(T² · L · d) cho attention. Mapper FLOPs:

```
FLOPs_map ≈ 2 · T · L_t · n_kv^t · (k · n_kv^s · d_h^s) · d_h^t
          = T × N_params_mapper
```

Tức là mapper là **tuyến tính theo T** với hệ số = số param mapper. Trong khi đó re-prefill của target là tuyến tính theo T với hệ số = số param target, **cộng thêm thành phần bậc hai T²**.

| Context | Nhận định |
|---|---|
| 1K | Mapper overhead có thể so sánh được với prefill; lợi ích mỏng |
| 4K–8K | Bắt đầu có lợi rõ; NVIDIA đo 2.7–25× |
| 32K | Vùng paper serve tới; thành phần T² của attention làm re-prefill đắt vọt |
| 128K–1M | **Vùng Cross-KV hấp dẫn nhất về lý thuyết** — nhưng đồng thời là vùng KV transfer size trở thành bottleneck (xem §7.4), và là vùng **chưa ai đo đạc** |

⚠️ **Cảnh báo:** RoPE-stripped content-space mapping được thiết kế để "extends to longer contexts by construction", nhưng calibration chỉ ở 1024 token và eval chỉ tới 32k. **Long-context stability là gap chưa được đo.**

### 7.3 So sánh Cross-KV vs mapper cost (đơn giản hoá)

Với Qwen3 14B→32B:
- Mapper: ~1.0–1.5B params → mỗi token tốn ~2–3 GFLOPs
- Re-prefill Qwen3-32B: mỗi token tốn ~64 GFLOPs (2×32B)

→ tỉ lệ ~20–30×, khớp với biên trên "25×" mà paper report.

### 7.4 Data movement — đây là phần thường bị bỏ qua

**Công thức KV size:**
```
Memory_KV = 2 × T × L × H_KV × d_head × bytes_per_element
```

**Số liệu đo thực tế từ literature:**

| Nguồn | Số liệu |
|---|---|
| HBM Is Not All You Need (2606.29986) | Qwen3-32B @ 8K ≈ **2 GB BF16** (1 GB BFP8). Qua RDMA 100 Gbps (~10 GB/s effective): **100–200 ms**, tức **10–15% prefill latency**, và bị tính **toàn bộ vào TTFT** nếu schedule ngây thơ |
| Prefill-as-a-Service (2604.15039) | MiniMax-M2.5 @ 32K sinh KV ở ~**60 Gbps** — vượt xa egress bandwidth cross-datacenter Ethernet thông thường |
| SwiftCache (2606.16135) | Qwen3-32B: 99% request có KV load time **<11 ms**, store **<6 ms** (trong node) |
| NVIDIA NVFP4 KV blog | NVFP4 KV cache: **up to 3× lower TTFT**, +20% cache hit rate vs FP8 |

**Ba insight về data movement:**

1. **Layer-wise pipelining là bắt buộc.** KV của layer ℓ sẵn sàng ngay khi attention của ℓ xong, và transfer của nó độc lập với layer ℓ+1... ⇒ có thể pipeline sau compute của các layer sau. DroidSpeak đo được cải thiện ~2× TTFT chỉ nhờ pipelining.

2. **E cache đắt hơn KV cache.** Với DroidSpeak, E cache lớn gấp **2×** KV cache ở Mistral-7B/Llama-3-8B, và gấp **4×** ở Llama-3.1-70B (do GQA nén KV nhưng không nén embedding). Đây là lý do partial recomputation có chi phí ẩn lớn.

3. **Cross-KV mapper có một lợi thế bandwidth tiềm ẩn chưa được khai thác:** nếu source model nhỏ hơn có KV cache nhỏ hơn (ít layer hơn), thì transfer KV_S rồi map tại đích **rẻ hơn** transfer KV_T. Nhưng nếu mapper weights (4–12GB) phải nằm ở phía nhận, thì nó chiếm HBM cạnh tranh với KV cache. **Trade-off này chưa được paper nào phân tích.** → research gap.

### 7.5 Serving engine integration status

| Engine | Cross-model KV support | Bằng chứng |
|---|---|---|
| **vLLM** | ❌ general; ✅ hẹp cho aLoRA | aLoRA paper mở rộng vLLM với base-aligned block hashing + activation-aware masking. LatentMAS phải "modify the partial inner package inside vLLM backend" vì "vLLM does not officially support modifying KV-cache". DroidSpeak implement qua vLLM + LMCache |
| **LMCache** | ⚠️ dùng làm KV store cho DroidSpeak | DroidSpeak built trên LMCache 0.1.4 |
| **SGLang** | Not found | RadixAttention là same-model prefix caching |
| **TensorRT-LLM** | Not found | `I could not verify any cross-model KV transfer support in TensorRT-LLM.` |
| **NVIDIA Dynamo / NIXL** | Not found cho cross-model | Dynamo được nhắc trong DroidSpeak như một distributed inference system nhưng cho same-model KV transfer |
| **llama.cpp** | Not found | — |
| **HF Transformers** | ⚠️ dùng cho research prototype | LatentMAS khuyến nghị dùng HF backend để reproduce official results |

> **Về issue/PR cụ thể trong các engine:** `I could not verify specific open issues or pull requests for general cross-model KV cache transfer in vLLM, SGLang, or TensorRT-LLM from primary sources.`

---

## 8. Comparison Table

| Approach | Zero recompute | Different model size | Different architecture | Training needed | Quality | Speedup | Code |
|---|---:|---:|---:|---:|---:|---:|---|
| **Cross-Model KV Transfer (ridge)** | ✅ | ✅ (8B→70B) | ⚠️ same family, matched-KV | ❌ (closed-form) | 73–98% (4/6 pairs); 42–44% (2/6) | 2.7–25× vs re-prefill | ❌ |
| **Cross-Model KV Transfer (MLP)** | ✅ | ✅ | ⚠️ same family | ✅ (Adam) | >90% HellaSwag trên cả 4 pair test | Not reported | ❌ |
| **DroidSpeak** | ❌ (~11% layer) | ❌ | ❌ (identical arch) | ❌ (chỉ profiling) | Negligible loss | 1.7–3.1× prefill; 4× throughput | ✅ (vLLM+LMCache) |
| **C2C** | ❌ (receiver vẫn prefill) | ✅ | ✅ | ✅ | +6.4–14.2% vs single model | 2.0× vs T2T | ✅ |
| **LCF** | ⚠️ | ✅ | ✅ | ✅ | Not fully reported | ~4% adapter size of C2C | ❌ |
| **LatentAlign** | ✅ (prefix cache) | ✅ (4L vs 16L) | ✅ | ✅ | ≥ base model NLL | Not reported | ❌ |
| **IAM** | ❌ | ✅ | ⚠️ | Not reported | "no appreciable sacrifice" | +15% prefill | ✅ |
| **aLoRA cross-model reuse** | ✅ | ❌ (cùng base) | ❌ | ❌ | Lossless by design | 58× e2e, >100× TTFT | ✅ (vLLM) |
| **ProxyKV** | ❌ | ✅ (proxy+target) | ⚠️ | ✅ | LongBench Pareto-dominant | 3.21× prefill | ❌ |
| Prefix caching (baseline E) | ✅ | ❌ | ❌ | ❌ | Lossless | — | ✅ |

### Ranking

**1. Most practical today** → **DroidSpeak**. Đã tích hợp vLLM + LMCache, tested enterprise, negligible quality loss, không cần train gì. Đổi lại: chỉ same-architecture.

**2. Most promising research direction** → **Cross-Model KV Transfer (NVIDIA)**. Vì nó chứng minh được điều bất ngờ nhất: quan hệ cross-model KV **đủ tuyến tính để fit closed-form**. Đây mở ra cả một class phương pháp không cần GPU training.

**3. Best quality** → **Cross-Model KV Transfer + MLP mapper** (>90% HellaSwag retention trên cả 4 pair test, bao gồm 2 pair mà ridge fail). Nếu tính "quality mà không mất gì" thì **aLoRA** và **DroidSpeak** thắng, nhưng chúng không cross-scale.

**4. Lowest overhead** → **Cross-Model KV Transfer (ridge)** về training overhead (47–87 phút, không gradient). Nhưng về **inference memory** thì tệ nhất (4–12GB/pair) — nếu tính overhead theo memory thì **LCF** thắng (~4% adapter size của C2C).

**5. Most scalable** → **LatentAlign**. Kiến trúc shared latent space Σ khiến số adapter tăng **O(N)** thay vì O(N²), và họ chứng minh được **zero-shot generalization sang translation path chưa từng train**. Đây là property duy nhất đúng nghĩa "scalable" trong toàn literature.

**6. Most likely to integrate into vLLM** → **aLoRA cross-model prefix reuse**. Đã có implementation trong vLLM execution path. Kế đến là DroidSpeak (đã dùng vLLM+LMCache). Cross-Model KV Transfer của NVIDIA cần vLLM có public API để **ghi** KV cache từ ngoài — hiện chưa có.

---

## 9. Information-Theoretic Bottleneck

### 9.1 Phát biểu formal

So sánh hai Markov chain:

```
Chain A (transfer):   X ──► KV_S ──► K̂V_T ──► output
Chain B (native):     X ──► KV_T ──────────► output
```

Theo data processing inequality, với mọi mapping f (deterministic hoặc stochastic):

```
I(X ; f(KV_S))  ≤  I(X ; KV_S)
```

Do đó nếu `I(X ; KV_S) < I(X ; KV_T)`, thì **không mapping nào** có thể tái tạo đầy đủ thông tin mà KV_T mang. Mapping chỉ **tái sắp xếp** thông tin đã có, không **tạo ra** thông tin mới.

### 9.2 Bằng chứng thực nghiệm cho bottleneck này

**Bằng chứng #1 — C2C oracle (mạnh nhất):** sau khi map KV của Qwen3-4B vào không gian Qwen3-0.6B, transformed cache "occupies only a smaller subset of the target's space". C2C kết luận thẳng: "the source model's semantic information cannot fully cover the target's, **despite the source being larger**."

Chú ý cụm cuối: **ngay cả khi source LỚN HƠN**, nó vẫn không phủ được target space. Điều này cho thấy bottleneck **không thuần tuý là vấn đề capacity** — mà là vấn đề **các model encode context theo cách khác nhau về bản chất**. C2C củng cố bằng quan sát thứ hai: tập câu trả lời đúng của các model khác nhau có **overlap hạn chế**, dù accuracy tổng thể tương đương.

**Bằng chứng #2 — R² trần của linear mapping:** Qwen3 14B→32B, ngay cả với multi-source-layer, R² chỉ đạt **0.79 (keys) / 0.65 (values)**. Với 8B→32B, đỉnh heatmap single-source chỉ 0.65 (vs 0.81 cho 14B→32B). **Khoảng cách kích thước lớn hơn ⇒ R² thấp hơn** — chính xác là điều bottleneck dự đoán.

**Bằng chứng #3 — K dễ dự đoán hơn V:** chênh lệch head-averaged R² khoảng **~0.2** giữa K và V. Diễn giải: K quyết định *chú ý vào đâu* (structural, dễ chia sẻ giữa các model cùng family); V mang *nội dung được đọc ra* (semantic, model-specific).

**Bằng chứng #4 — LatentAlign, chiều ngược:** khi source **mạnh hơn** target, translation dễ hơn. LatentAlign quan sát "it takes less data to learn a translation from the stronger model's k-v cache space to the weaker model's". Đây là chiều large→small, và nó **rẻ hơn** small→large — nhất quán với bottleneck.

### 9.3 Nhưng — bottleneck KHÔNG phải là toàn bộ câu chuyện

Đây là điểm tinh tế nhất trong toàn bộ báo cáo.

Paper NVIDIA đo tương quan giữa các metric và HellaSwag retention trên **12 matched-KV pair evaluation từ 3 family**:

```
attention-output cosine  vs  retention  :  Pearson r = +0.57
R²                       vs  retention  :  Pearson r = −0.20
```

**R² có tương quan ÂM với chất lượng.** Điều này bác bỏ trực tiếp giả thuyết ngây thơ "R² cao ⇒ transfer tốt".

Giải thích của paper: **error placement, not error magnitude, determines per-pair retention.** Cùng một mức sai số Frobenius, nếu nó nằm trong subspace mà attention nhạy cảm thì phá huỷ output; nếu nằm trong hướng attention-irrelevant thì vô hại.

Bằng chứng củng cố: MLP **không giảm error tổng thể so với ridge** (cùng MSE objective!) nhưng đạt +37pp HellaSwag trên pair khó — vì nó **redistribute residual error away from attention-sensitive subspaces**.

### 9.4 Trả lời trực tiếp các câu hỏi ở §9 của brief

**"79% variance explained nghĩa là gì?"**
```
R² = 1 − ‖KV_T − K̂V_T‖² / ‖KV_T − mean(KV_T)‖²
```
Nghĩa là: 79% phương sai (quanh mean) của target KV được giải thích bởi linear function của source KV, tính head-averaged, layer-averaged, trên calibration distribution.

**"Nó có tương đương 79% model quality không?"**
**Không, hoàn toàn không.** Bằng chứng thực nghiệm: cùng R² ≈ 0.79 cho Qwen3 14B→32B tương ứng **97.6% retention**, tức là R² *đánh giá thấp* chất lượng ở pair này. Ngược lại, các pair Ministral có R² không tệ nhưng retention chỉ 41–44%. **Không có ánh xạ đơn điệu nào giữa R² và quality** — và tương quan đo được thậm chí là âm (r = −0.20).

**"Nó cho biết giới hạn gì của linear mapping?"**
- 21% residual không nhất thiết là "thông tin bị mất" — có thể là nonlinear structure mà MLP bắt được.
- Nhưng phần residual mà **cả ridge lẫn MLP đều không bắt được** thì mới thực sự là information bottleneck.
- Bằng chứng phân biệt: MLP đưa được **cả 4 pair test lên >90%** ⇒ với các family này, phần lớn residual là **nonlinearity chứ không phải information loss**. Đây là tin tốt bất ngờ.

**"Remaining 21% có chứa thông tin quan trọng không?"**
Có — nhưng **có chọn lọc theo task**. GSM8K (multi-step CoT reasoning) rơi xuống 1.6–18.2% ở các pair mà HellaSwag vẫn giữ 58–94%. Residual chứa chính xác thứ mà **long-horizon reasoning** cần. Đây là chữ ký của **error accumulation**: sai số nhỏ trên mỗi token, khuếch đại qua chuỗi suy luận nhiều bước.

---

## 10. Existing Solutions to the Bottleneck

| # | Hướng | Đã có ai làm? | Bằng chứng |
|---|---|---|---|
| **A** | Distillation trước deployment (train small model để KV transferable) | ❌ **CHƯA AI LÀM** ở scale LLM | Không tìm thấy paper nào có loss `L_LM + λ·L_KV-align` để tạo transferable KV |
| **B** | Joint training source + target + mapper | ⚠️ một phần | LatentAlign train adapter nhưng **freeze cả hai model**; C2C cũng freeze cả hai |
| **C** | Auxiliary latent state Z | ✅ | **LatentAlign** (shared space Σ), **LCF** (low-dim latent bottleneck) |
| **D** | Nonlinear mapper | ✅ | NVIDIA MLP (+24.3 → +36.8 pp); C2C neural fuser; LatentAlign cross-attention adapter |
| **E** | Low-rank residual correction `KV_T ≈ W·KV_S + ΔKV` | ⚠️ gián tiếp | LISA (2408.01890) dùng low-rank matrices xấp xỉ *khác biệt* attention weight giữa layer — nhưng là intra-model. **Chưa ai làm cross-model.** |
| **F** | Partial target recomputation | ✅ **Trưởng thành nhất** | DroidSpeak (11% critical layers, contiguous group); ProxyKV |
| **G** | Confidence-based routing (reuse → reject → recompute) | ❌ **CHƯA AI LÀM** | Không tìm thấy work nào có runtime confidence gate cho cross-model KV |

**Ba gap nổi bật nhất từ bảng này: A, E, G.** Đây cũng là nơi tôi đặt các đề xuất research ở §12.

Một quan sát bổ sung về hướng C: LatentAlign phát hiện rằng **reconstruction loss là không cần thiết** khi đã có suffix LM loss — và thậm chí **có hại** (Recon Weight 0.0 → 3.406; 0.5 → 3.531; 1.0 → 3.626). Điều này rất quan trọng vì nó nói: **đừng tối ưu ‖KV_T − K̂V_T‖²**. Nó nhất quán hoàn hảo với phát hiện R² r=−0.20 của NVIDIA. Hai group độc lập, hai phương pháp khác nhau, cùng một kết luận: **reconstruction fidelity là proxy sai.**

---

## 11. Open Problems

### 11.1 Đã được thừa nhận trong literature

| Vấn đề | Trạng thái |
|---|---|
| **Small → large information bottleneck** | Được C2C ghi nhận qua t-SNE; NVIDIA ghi nhận qua R² gap; **chưa ai định lượng formal** |
| **Layer alignment** | NVIDIA giải bằng top-k R² selection — **empirical, không có lý thuyết**. Vì sao k=8 cho Qwen3 nhưng k=all cho Ministral 3B→8B? Not explained |
| **Head alignment** | Né tránh bằng cách feed all-source-heads vào mỗi target head. **Chưa ai giải head correspondence tường minh** |
| **Different d_head** | Formula W ∈ R^{...×d_h^t} về lý thuyết cho phép, nhưng **chưa có kết quả thực nghiệm nào** |
| **Different GQA configs (n_kv^s ≠ n_kv^t)** | **Hoàn toàn chưa được giải.** Đây là điều kiện "matched-KV" bị vi phạm |
| **RoPE compatibility** | ✅ **ĐÃ GIẢI** — content-space mapping với R_Θ^{-1} exact (orthogonal). Một trong ít vấn đề có solution sạch |
| **Different positional embeddings** (ALiBi, NoPE, learned) | Not addressed |
| **Different tokenizer / vocabulary** | NVIDIA né bằng giả định same-family same-tokenizer. LatentAlign ghi chú: **suffix LM loss áp dụng được cho vocab khác nhau, reconstruction loss thì không** — một lối thoát tiềm năng |
| **Different architecture (Llama → Qwen)** | **Chưa ai chứng minh zero-recompute** |
| **Mapper generalization** | Domain sensitivity đo được: CodeAlpaca calibration → −5.24pp HellaSwag |
| **Mapper training cost** | 47–87 phút/pair trên 8×H100. Với N model thì O(N²) cặp ⇒ **không scale** |
| **Long-context stability** | Calibrate ở 1024 token, eval ≤32k. **1M token: hoàn toàn chưa test** |
| **Error accumulation during decode** | Bằng chứng gián tiếp mạnh: GSM8K collapse. **Chưa ai đo formal theo số token đã decode** |
| **KV distribution drift** | DroidSpeak thừa nhận data drift làm hỏng profiling config; đề xuất periodic re-profiling |
| **Per-token vs whole-sequence mapping** | NVIDIA fit ở **token level** (mỗi token = 1 observation), không có cross-token context. **Chưa ai thử sequence-level mapper** |
| **Cross-GPU KV transfer bandwidth** | Đo được ở same-model (100–200ms cho 2GB qua 100Gbps RDMA); **chưa ai đo cho cross-model mapper pipeline** |
| **Model version compatibility** | Not addressed — nếu vendor release Qwen3.1, mapper Qwen3 có còn dùng được? |

### 11.2 Ba open problem tôi cho là quan trọng nhất mà literature CHƯA nêu

**(i) Mapper memory vs KV cache memory trade-off.** Mapper 4–12GB nằm trên HBM của node đích, cạnh tranh trực tiếp với KV cache pool. Với 12 directed pair trong hệ 4-tier, có thể lên ~144GB. **Không paper nào phân tích điểm hoà vốn này.** Có thể tồn tại chế độ mà re-prefill rẻ hơn vì mapper ăn mất KV cache capacity ⇒ giảm hit rate ⇒ tăng TTFT trung bình.

**(ii) Không có metric predictive trước khi fit.** Hiện tại quy trình là: fit mapper (47–87 phút) → eval downstream → phát hiện pair này fail. Attention-output cosine chỉ r=+0.57, tức R²≈0.32 — **giải thích được 1/3 variance**. Cần một **a-priori pairability score** rẻ.

**(iii) Cross-model transfer chưa được đo trên hybrid attention.** Qwen-style GDN, Mamba hybrid, sliding-window/global mix (Gemma). NVIDIA nói rõ: "All families use dense full-attention, so every target layer receives mapped KV." **Với hybrid attention, layer linear-attention không có KV cache theo nghĩa thông thường — khái niệm transfer cần định nghĩa lại.**

---

## 12. New Research Opportunities

### Idea 1 — Distillation-aware Cross-KV (DA-CKV)

Train small model từ đầu (hoặc continual-pretrain) với loss:

```
L = L_LM  +  λ · L_KV-align
L_KV-align = Σ_{l,h} ‖ W_frozen · K_S^{(l,h)} − K_T^{(l',h)} ‖²  (+ tương tự cho V)
```

với target model frozen. **Twist quan trọng:** dựa trên phát hiện r=−0.20 của NVIDIA và ablation reconstruction-loss của LatentAlign, **KHÔNG dùng MSE**. Thay bằng **attention-output alignment loss**:

```
L_KV-align = Σ ‖ softmax(Q_T K̂_T^T/√d) V̂_T  −  softmax(Q_T K_T^T/√d) V_T ‖²
```

tức là align *đầu ra của attention*, không align KV.

| Tiêu chí | Đánh giá |
|---|---|
| Novelty | **Cao** — không tìm thấy work nào làm điều này |
| Technical difficulty | Trung bình-cao (cần train small model) |
| GPU requirements | **Rất cao** — pretrain/continual-pretrain một model 1–8B |
| Dataset | Pretraining-scale corpus |
| Likelihood of publishable result | Cao (kể cả negative result cũng có giá trị) |
| Systems impact | Rất cao nếu vendor adopt — biến "transferable KV" thành design property |

### Idea 2 — Universal KV Latent Space với routing-aware training

```
Model A KV ──T_A→Σ──► Z (universal) ──T_Σ→B──► Model B KV
```

Xây trên LatentAlign nhưng: (a) scale lên model 7B–70B thực (LatentAlign chỉ 100–400M), (b) thay reconstruction loss bằng suffix-LM loss (đã chứng minh tốt hơn), (c) khai thác extensibility đã chứng minh — thêm model chỉ cần 2 adapter, zero-shot với path chưa train.

| Tiêu chí | Đánh giá |
|---|---|
| Novelty | Trung bình (LatentAlign đã có framework), cao ở phần scale |
| Technical difficulty | Cao |
| GPU requirements | Cao (adapter ~¼ base model size × N model) |
| Likelihood | Cao — LatentAlign đã chứng minh feasibility ở small scale |
| Systems impact | **Cao nhất về scaling** — O(N) thay O(N²) |

### Idea 3 — Hierarchical / Chained Cross-KV (7B → 14B → 32B → 70B)

Câu hỏi nghiên cứu: transfer **chuỗi** có tích luỹ lỗi tệ hơn transfer **trực tiếp** 7B→70B không?

Có bằng chứng gián tiếp ủng hộ chuỗi: NVIDIA thấy R² đỉnh cao hơn cho 14B→32B (0.81) so với 8B→32B (0.65) — **model càng gần thì mapping càng dễ**. Vậy có thể `7B→14B→32B` tốt hơn `7B→32B` nếu mỗi bước gần nhau đủ để bù lỗi tích luỹ.

Cần thiết kế: **error budget analysis** per hop, và so sánh với direct mapper.

| Tiêu chí | Đánh giá |
|---|---|
| Novelty | Trung bình-cao |
| Technical difficulty | **Thấp** (chỉ cần compose các mapper đã có) |
| GPU requirements | Trung bình (fit 3 mapper thay 1) |
| Likelihood | **Rất cao** — thí nghiệm sạch, kết quả rõ ràng dù âm hay dương |
| Systems impact | Cao — cho phép mapper chỉ giữa các tier kề nhau, giảm O(N²) → O(N) |

### Idea 4 — KV Residual Reconstruction với Adaptive Rank

```
K̂V_T = W · KV_S  +  Δ(KV_S)
```
với Δ là low-rank correction, và **rank được cấp phát theo attention-sensitivity của từng (layer, head)** đo bằng attention-output cosine.

Động cơ: NVIDIA cho thấy MLP thắng ridge **chỉ vì redistribute error**, không phải vì giảm error. Vậy: thay vì MLP đắt ở mọi head, chỉ đầu tư capacity vào head nhạy cảm. Đây là **compute-optimal error placement**.

| Tiêu chí | Đánh giá |
|---|---|
| Novelty | **Cao** — kết hợp phát hiện error-placement với low-rank adaptivity |
| Technical difficulty | Trung bình |
| GPU requirements | **Thấp** (fit từ mapper ridge có sẵn) |
| Dataset | Nhỏ (~500 sequence như NVIDIA) |
| Likelihood | Cao |
| Systems impact | Cao — giảm mapper size từ 4–12GB xuống đáng kể |

### Idea 5 — Adaptive Cross-KV với Confidence Gating

Runtime quyết định per (token, layer, head): transfer hay recompute.

```
score(l,h,t) = f( ‖residual‖, attention-sensitivity(l,h), position t )
if score < θ:  transfer
else:          recompute layer group
```

Đây là **hợp nhất DroidSpeak (partial recompute) với NVIDIA mapper (transform)** — hiện tại hai hướng này hoàn toàn tách biệt trong literature. DroidSpeak recompute layer nhưng không transform; NVIDIA transform nhưng không recompute gì.

**Hybrid tự nhiên:** map KV bằng ridge cho mọi layer → dùng attention-output cosine để phát hiện layer mà mapping kém → chỉ recompute nhóm layer đó (contiguous, theo bài học DroidSpeak về transition point).

| Tiêu chí | Đánh giá |
|---|---|
| Novelty | **Rất cao** — chưa ai kết hợp |
| Technical difficulty | Trung bình-cao (cần serving integration) |
| GPU requirements | Trung bình |
| Likelihood | **Rất cao** — gần như chắc chắn cải thiện Pareto frontier so với cả hai baseline |
| Systems impact | **Cao nhất** — giải quyết trực tiếp 2/6 pair fail của NVIDIA |

### Idea 6 (bonus) — A-priori Pairability Score

Dự đoán **trước khi fit** liệu một cặp model có transfer được không, dùng các signal rẻ: CKA/CCA giữa KV distribution trên 1K token, spectral overlap, attention-output cosine sau một fit thô rank thấp.

Rẻ, hữu ích ngay, và giải quyết open problem (ii) ở §11.2. Novelty trung bình, difficulty thấp, likelihood cao.

---

## 13. Potential vLLM Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM Engine (Model T)                     │
│                                                              │
│  Scheduler ──► CrossModelKVConnector (NEW)                   │
│                   │                                          │
│                   ├── lookup(context_hash, src_model_id)     │
│                   ├── fetch KV_S  (LMCache / NIXL / RDMA)    │
│                   ├── KVMapper.apply(KV_S) ──► KV_T          │
│                   │      ├── strip_rope(K_S, Θ_s)            │
│                   │      ├── gather top-k source layers      │
│                   │      ├── batched GEMM per (l,h)          │
│                   │      └── apply_rope(K̂_T, Θ_t)            │
│                   ├── [optional] confidence gate             │
│                   │      └── partial_prefill(critical_grp)   │
│                   └── write into PagedAttention blocks       │
│                                                              │
│  KVCacheManager ◄── needs NEW public write API               │
└─────────────────────────────────────────────────────────────┘
```

**Những gì vLLM đang thiếu (verified):**

1. **Public API để ghi KV cache từ bên ngoài.** Bằng chứng trực tiếp: LatentMAS phải "modify the partial inner package inside vLLM backend". Đây là blocker số 1.

2. **Cross-model block hashing.** Prefix cache hash hiện tại gắn với model. aLoRA paper đã giải một trường hợp hẹp bằng "base-aligned block hashing" — pattern này có thể tổng quát hoá.

3. **Mapper weight management.** 4–12GB per pair cần một memory pool riêng, có thể offload/paging như KV cache.

4. **Layer-wise streaming của mapper.** Bài học từ DroidSpeak (CUDA stream riêng cho transmission) và HBM-Is-Not-All-You-Need (layer ℓ's KV độc lập với ℓ+1) ⇒ mapper nên chạy per-layer và overlap với transfer.

**Đường đi khả thi nhất (theo thứ tự):**
```
Phase 1: mở rộng LMCache thành cross-model KV store (đã có precedent DroidSpeak)
Phase 2: KVMapper như một plugin, dùng offline-fitted weights (safetensors)
Phase 3: public KVCacheManager write API trong vLLM
Phase 4: connector + scheduler integration (router-aware)
```

---

## 14. Recommended Experiments

**E1 — Reproduce NVIDIA ridge mapper (baseline bắt buộc).**
Qwen3 8B→32B, k=12, 500 FineWeb-Edu seq @1024, λ=0.01, content-space. Target: Avg retention ≈ 87.5%. Nếu không reproduce được thì mọi thứ sau đều vô nghĩa. Chi phí: ~1 node 8×H100 × 2 giờ.

**E2 — Đo error accumulation theo decode length.**
Chưa ai làm. Đo retention như hàm của số token đã sinh (1, 8, 32, 128, 512, 2048). Giả thuyết: divergence gần như tuyến tính-log, giải thích GSM8K collapse. Rẻ, novel, có thể publish riêng.

**E3 — Long-context stress test.**
Calibrate ở 1024, eval ở 4K/16K/64K/128K. Kiểm chứng claim "extends to longer contexts by construction". Đây là gap lớn nhất chưa ai lấp.

**E4 — Mismatched GQA.**
Thử pair vi phạm matched-KV (n_kv^s ≠ n_kv^t). Về mặt formula W vẫn hợp lệ. Nếu nó chạy được thì mở rộng đáng kể phạm vi áp dụng; nếu không thì xác lập được biên giới rõ ràng.

**E5 — Hybrid mapper + selective recompute (Idea 5).**
Ministral 8B→14B (pair fail tệ nhất, 41.6%). Map toàn bộ bằng ridge, rồi recompute contiguous group 4/8/12 layer chọn theo attention-output cosine. Đo Pareto: retention vs prefill latency. **Đây là thí nghiệm có expected value cao nhất.**

**E6 — Ablation: MSE vs attention-output loss cho MLP mapper.**
Trực tiếp test giả thuyết "reconstruction là proxy sai". Cùng kiến trúc MLP, đổi loss. Nếu attention-output loss thắng rõ, đó là kết quả gọn và có tác động lớn.

**E7 — Chained transfer (Idea 3).**
Qwen3 8B→14B→32B vs 8B→32B trực tiếp. Đo cả retention và tổng mapper size.

**E8 — Mapper memory vs KV pool trade-off.**
Serving simulation: 4-tier routing, đo goodput với mapper on-HBM vs re-prefill, quét theo QPS và context length. Trả lời câu hỏi kinh tế thật.

---

## 15. Complete Paper List

### Category A — True cross-model KV transfer
1. Heo, Shafipour, Zhao, Golub, Kamani, Borkar, Chandran, Zardoshti, Rouhani (NVIDIA). *Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse.* arXiv:2608.03893, 2026.
2. Dery, Yahav, Prior, Feng, Shen, Szlam (Google DeepMind). *Latent Space Communication via K-V Cache Alignment.* arXiv:2601.06123, 2026.
3. Rossi et al. (Columbia). *Latent Cache Flow: Model-to-Model Communication Without Text.* arXiv:2605.22863, 2026.

### Category B — Cross-model representation transfer
4. Fu, Min, Zhang, Yan, Dai, Ouyang, Wang. *Cache-to-Cache: Direct Semantic Communication Between Large Language Models.* arXiv:2510.03215. ICLR 2026.
5. Zhao, Li, Zhao. *IAM: Efficient Inference through Attention Mapping between Different-scale LLMs.* arXiv:2507.11953. ACL 2025.
6. *LatentMAS: Latent Collaboration in Multi-Agent Systems.* ICML 2026 Spotlight.
7. Liu et al. *When Hidden States Drift: Can KV Caches Rescue Long-Range Speculative Decoding?* arXiv:2604.26412, 2026.
8. Du et al. *GliDe with a CaPE.* (draft cross-attends target KV), 2024.
9. Yang et al. *LongSpec: Long-Context Lossless Speculative Decoding.* arXiv:2502.17421, 2025.
10. Li et al. *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* arXiv:2401.15077.
11. Cai et al. *Medusa.* arXiv:2401.10774.

### Category C — Shared / compatible KV architecture
12. Li, Greenewald, Parnell, Azizan (MIT/IBM). *Efficient Multi-Adapter LLM Serving via Cross-Model KV-Cache Reuse with Activated LoRA.* arXiv:2512.17910.
13. *KV Cache Transform Coding for Compact Storage in LLM Inference.* arXiv:2511.01815. ICLR 2026.
14. Mu et al. *Cross-layer Attention Sharing for Pre-trained Large Language Models (LISA).* arXiv:2408.01890.
15. Brandon et al. *Reducing Transformer Key-Value Cache Size with Cross-Layer Attention.* 2024.
16. Wu & Tu. *Layer-Condensed KV Cache for Efficient Inference of LLMs.* ACL 2024.

### Category D — Partial recomputation
17. Liu, Huang, Yao, Feng, Gu, Du, Li, Cheng, Jiang, Lu, Musuvathi, Choukse. *DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving.* arXiv:2411.02820.
18. *ProxyKV: Cross-Model Proxy Pruning for Efficient Long-Context LLM Inference.* arXiv:2605.16360.
19. Yao et al. *CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion.* 2024.

### Category E — Same-model reuse (baseline)
20. Kwon et al. *Efficient Memory Management for LLM Serving with PagedAttention (vLLM).* SOSP 2023.
21. Zheng et al. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS 2024.
22. Gim et al. *Prompt Cache: Modular Attention Reuse for Low-Latency Inference.* 2024.
23. Liu et al. *CacheGen: KV Cache Compression and Streaming.* 2024.
24. Lu et al. *TurboRAG.* 2025. / Yang et al. *KVLink.* 2025.
25. *SparseX: Efficient Segment-Level KV Cache Sharing for Interleaved LLM Serving.* arXiv:2606.01751.
26. *CacheTune: Adaptive KV Cache Reuse for Fast Long-Context LLM Serving.* arXiv:2605.24022.
27. *SwiftCache.* arXiv:2606.16135.

### Category F — KV compression
28. *xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction.* arXiv:2503.18893. ICML 2026.
29. Liu et al. *KIVI.* arXiv:2402.02750. / Li et al. *SnapKV.* NeurIPS 2024. / Cai et al. *PyramidKV.* / Zhang et al. *H2O.*
30. Xiao et al. *StreamingLLM.* arXiv:2309.17453. / *DuoAttention.* arXiv:2410.21465.

### Representation alignment / model stitching (nền tảng toán học)
31. Huh et al. *The Platonic Representation Hypothesis.* 2024.
32. Moschella et al. *Relative Representations Enable Zero-Shot Latent Space Communication.* arXiv:2209.15430.
33. Lähner & Möller. *On the Direct Alignment of Latent Spaces.* UniReps 2024.
34. Jha, Zhang, Shmatikov, Morris. *Harnessing the Universal Geometry of Embeddings.* arXiv:2505.12540.
35. Maiorca et al. *Latent Space Translation via Semantic Alignment.* NeurIPS 2023.
36. Ainsworth, Hayase, Srinivasa. *Git Re-Basin.* arXiv:2209.04836.
37. Chen et al. *bert2bert: Towards Reusable Pretrained Language Models.* arXiv:2110.07143.

### Systems / data movement
38. *HBM Is Not All You Need: Efficient Disaggregated LLM Serving across Memory-heterogeneous Accelerators.* arXiv:2606.29986.
39. *Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter.* arXiv:2604.15039.
40. Zhong et al. *DistServe.* OSDI 2024. / Patel et al. *Splitwise.* 2024.
41. *Serving Large Language Models on Huawei CloudMatrix384.* arXiv:2506.12708.

---

## 16. Complete GitHub Repository List

| Repo | Work | Verified |
|---|---|---|
| https://github.com/thu-nics/C2C | Cache-to-Cache (ICLR'26) | ✅ official |
| https://github.com/QQQ-yi/IAM | IAM (ACL 2025) | ✅ stated in paper |
| https://github.com/Gen-Verse/LatentMAS | LatentMAS (ICML'26 Spotlight) | ✅ official |
| https://github.com/abdelfattah-lab/xKV | xKV (ICML'26) | ✅ stated in paper |
| https://github.com/LMCache/LMCache | KV cache layer (DroidSpeak dùng) | ✅ |
| https://github.com/vllm-project/vllm | Serving engine | ✅ |
| https://github.com/vllm-project/production-stack | vLLM Production Stack (DroidSpeak deploy) | ✅ |
| https://github.com/vllm-project/aibrix | ByteDance AIBrix | ✅ |
| https://github.com/ai-dynamo/dynamo | NVIDIA Dynamo | ✅ |
| https://github.com/whyNLP/LCKV | Layer-Condensed KV + CLA | ✅ |
| https://github.com/Zefan-Cai/PyramidKV | PyramidKV | ✅ |
| https://huggingface.co/collections (C2C fusers) | C2C fuser weights | ✅ per README |
| — | **Cross-Model KV Transfer (NVIDIA)** | ❌ **Not found** |
| — | LatentAlign (Google DeepMind) | ❌ Not found |
| — | Latent Cache Flow | ❌ Not found (patent pending) |
| — | DroidSpeak (public URL) | ❌ anonymized in arXiv version |
| — | ProxyKV | ❌ Not found |

---

## 17. References (primary sources)

1. https://arxiv.org/abs/2608.03893 — Cross-Model KV Cache Transfer in LLM Families (NVIDIA), full text: https://arxiv.org/html/2608.03893v1
2. https://arxiv.org/abs/2510.03215 — Cache-to-Cache (C2C); project: https://fuvty.github.io/C2C_Project_Page/ ; code: https://github.com/thu-nics/C2C ; OpenReview: https://openreview.net/forum?id=LeatkxrBCi
3. https://arxiv.org/abs/2601.06123 — Latent Space Communication via K-V Cache Alignment (Google DeepMind)
4. https://arxiv.org/abs/2411.02820 — DroidSpeak (UChicago + Microsoft)
5. https://arxiv.org/abs/2507.11953 — IAM (ACL 2025); code: https://github.com/QQQ-yi/IAM
6. https://arxiv.org/pdf/2605.22863 — Latent Cache Flow (Columbia)
7. https://arxiv.org/pdf/2512.17910 — Cross-Model KV-Cache Reuse with Activated LoRA (MIT + IBM Research)
8. https://arxiv.org/pdf/2605.16360 — ProxyKV
9. https://arxiv.org/pdf/2511.01815 — KV Cache Transform Coding (ICLR 2026)
10. https://arxiv.org/pdf/2503.18893 — xKV (ICML 2026)
11. https://arxiv.org/abs/2604.26412 — When Hidden States Drift (speculative decoding KV reuse)
12. https://arxiv.org/html/2502.17421v2 — LongSpec
13. https://arxiv.org/abs/2408.01890 — Cross-layer Attention Sharing (LISA)
14. https://arxiv.org/pdf/2606.29986 — HBM Is Not All You Need (KV transfer cost over RDMA)
15. https://arxiv.org/html/2604.15039v1 — Prefill-as-a-Service (KV throughput)
16. https://arxiv.org/pdf/2606.16135 — SwiftCache
17. https://arxiv.org/pdf/2606.01751 — SparseX
18. https://arxiv.org/pdf/2605.24022 — CacheTune
19. https://arxiv.org/pdf/2506.12708 — Huawei CloudMatrix384 (RDMA KV transfer architecture)
20. https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/ — NVIDIA NVFP4 KV cache
21. https://github.com/Gen-Verse/LatentMAS — LatentMAS (vLLM KV-cache limitation note)

---

## Phụ lục: Tên gọi khác nhau cho cùng technique

| Technique | Các tên gọi trong literature |
|---|---|
| Cross-model KV transfer | "cross-model KV cache transfer" (NVIDIA), "cross-model KV sharing", "inter-model KV cache reuse", "KV cache translation", "Cross-KV" (tên phi chính thức trong thảo luận cộng đồng) |
| C2C | "Cache-to-Cache", "C2C", "Rosetta" (tên repo) |
| LatentAlign | "LatentAlign" (cách NVIDIA gọi), tên chính thức: "Latent Space Communication via K-V Cache Alignment" |
| Sender/Receiver | DroidSpeak dùng "sender/receiver"; C2C dùng "sharer/receiver"; NVIDIA dùng "source/target" |

> **Lưu ý về thuật ngữ "Cross-KV":** brief hỏi cụ thể về "NVIDIA Cross-KV". Paper NVIDIA tự gọi technique là **"cross-model KV cache transfer"**, không phải "Cross-KV". `I could not verify the existence of an official NVIDIA product or codename "Cross-KV" from primary sources.`

---

## Trả lời cuối cùng

> **Tính đến hiện tại (2026-08), state-of-the-art thực sự của việc transferring KV cache giữa hai LLM khác nhau mà không recompute prefill là gì?**

**SOTA:** per-head closed-form ridge regression mapper của NVIDIA (arXiv:2608.03893), với ba thành phần: (1) top-k cross-layer source selection, (2) per-(layer, head) ridge với λ=0.01 fit trên ~128K calibration token, (3) RoPE-stripped content-space mapping. Đạt 73–98% retention trên 4/6 matched-KV pair, 2.7–25× nhanh hơn re-prefill, stable qua multi-turn handoff.

> **Technique nào đã được chứng minh bằng experiment?**
- Zero-recompute cross-scale transfer: ✅ NVIDIA ridge/MLP mapper (6 pair, 3 family, 5 benchmark + PPL + CoQA)
- Partial-recompute same-arch: ✅ DroidSpeak (8 pair, 6 dataset, deploy K8s)
- KV fusion (không skip prefill): ✅ C2C (4 model family)
- Shared latent space: ✅ LatentAlign (nhưng chỉ ở model 100–400M)
- Compatible-by-design: ✅ aLoRA (58× e2e, >100× TTFT)

> **Technique nào có source code?**
C2C (thu-nics/C2C), IAM (QQQ-yi/IAM), LatentMAS (Gen-Verse/LatentMAS), aLoRA (vLLM extension), DroidSpeak (vLLM+LMCache, URL anonymized). **SOTA của NVIDIA: không có code.**

> **Những bottleneck nào vẫn chưa được giải quyết?**
1. **Matched-KV constraint** — khác n_kv hoặc d_head: chưa có kết quả nào
2. **Cross-family** (Llama→Qwen, khác tokenizer): chưa ai chứng minh zero-recompute
3. **Reasoning collapse** — GSM8K rơi xuống 1.6–18.2% ngay cả khi HellaSwag giữ >90%
4. **R² là proxy sai** (r=−0.20) và chưa ai có metric predictive tốt (attention-output cosine chỉ r=+0.57)
5. **Mapper 4–12GB/pair** — không scale O(N²) qua nhiều tier
6. **Long context >32k** — hoàn toàn chưa đo
7. **Hybrid attention** (Mamba/GDN/linear attention) — khái niệm transfer chưa được định nghĩa
8. **Không có confidence-based routing** — không có cơ chế phát hiện transfer hỏng ở runtime
9. **vLLM không có public API để ghi KV cache** — blocker kỹ thuật cho mọi integration
10. **Không có work nào về distillation-aware transferable KV** — hướng có tiềm năng nhất nhưng chưa ai chạm tới
