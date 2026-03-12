# llm_stats.py — LLM Architecture Stats Tool

Fetches and compares architectural statistics for any open LLM on HuggingFace **without downloading model weights** — only `config.json` (and optionally `tokenizer_config.json`) are fetched, both small JSON files.

Useful for quickly understanding a new model release without reading the paper.

**Only dependency:** `pip install requests` (no PyTorch, no `huggingface_hub`)

---

## Quick Start

```bash
# Compare several models side-by-side
python llm_stats.py Qwen/Qwen3-1.7B meta-llama/Llama-3.2-1B mistralai/Mistral-7B-v0.3

# VRAM estimate at a realistic sequence length (not the full 128K max context)
python llm_stats.py --seq-len 4096 Qwen/Qwen3-1.7B

# Export to JSON or CSV
python llm_stats.py --format json Qwen/Qwen3-1.7B > stats.json
python llm_stats.py --format csv model1 model2 > comparison.csv

# Debug an unknown model: print raw config.json and exit
python llm_stats.py --dump-raw new-org/new-model

# Access gated models (e.g. Llama, Gemma)
python llm_stats.py --token $HF_TOKEN meta-llama/Meta-Llama-3-8B

# Pin to a specific commit or tag
python llm_stats.py --revision v1.0 mistralai/Mistral-7B-v0.1
```

---

## Output

The script prints four sections in order.

### Section 1 — Architecture Table

One row per model with the core structural fields:

| Column | Description |
|---|---|
| Model | HuggingFace model ID |
| Type | `model_type` from config (e.g. `llama`, `qwen3`, `mistral`) |
| Tokenizer | `tokenizer_class` from `tokenizer_config.json` |
| Vocab | Vocabulary size |
| Context | Max position embeddings (e.g. `128K`, `40K`) |
| Layers | Number of transformer layers |
| Heads | Number of query attention heads |
| KV-Heads | Number of key/value heads (same as Heads for MHA; fewer for GQA/MQA) |
| HeadDim | Dimension per head (`hidden_size / num_attention_heads`, or explicit) |
| FFN-Size | Intermediate (hidden) size in the FFN block |
| FFN-Ratio | `FFN-Size / hidden_size` — how much the FFN expands the residual stream |
| Attn-Type | `MHA` (Multi-Head), `MQA` (Multi-Query), or `GQA-Nx` (Grouped-Query, N = Q/KV ratio) |
| Activation | FFN activation function (e.g. `silu`, `gelu`) |
| QK-Norm | Whether per-head query/key normalization is applied (Qwen3, Gemma3) |
| RoPE-Base | `rope_theta` — higher values correlate with longer-context training |
| Dtype | Native parameter dtype (e.g. `bfloat16`) |
| KV-Cache/tok | KV cache memory per token (bf16): `2 × kv_heads × head_dim × layers × 2 bytes` |

### Section 2 — Parameter Breakdown

Shows how parameters are distributed across components:

| Column | Description |
|---|---|
| Total | Total parameter count |
| Active | Active params per forward pass (same as Total for dense models; less for MoE) |
| Embed (%) | Token embedding table share of total |
| Attn-Total (%) | All attention layers combined |
| FFN-Total (%) | All FFN blocks combined |
| Output-Head (%) | LM head (0% if `tie_word_embeddings=True`) |

### Section 3 — VRAM Estimates

| Column | Description |
|---|---|
| Weights-bf16/int8/int4 | Model weight memory at each quantization level |
| KV@MaxCtx | KV cache at the model's full context length (worst case) |
| KV@SeqLen | KV cache at `--seq-len` (same as KV@MaxCtx if flag not passed) |
| Total-bf16/int8/int4 | Weights + KV@SeqLen + 1 GB overhead |
| Min-GPU | Smallest standard GPU that fits the int4 total |

KV cache is always counted at bf16 even when weights are quantized — this matches standard inference practice (quantizing the KV cache is uncommon in production).

GPU tiers: 4 GB → 8 GB → 12 GB → 16 GB → 24 GB → 40 GB → 48 GB → 80 GB → 141 GB.

### Section 4 — Model Insights (Narrative)

Plain-English deductions per model, for example:

```
── Qwen/Qwen3-1.7B ─────────────────────────────────────────────────────────
  Structure:  28 layers × 61.7M params/layer = 1.73B layer params
  Budget:     Embed 18% | Attention 37% | FFN 45% | LM-head 0% (tied)
  Attention:  GQA-2x — 16 Q-heads share 8 KV-heads; KV cache 2× smaller than MHA
              head_dim=128; each layer stores 2 × 8 × 128 = 2048 values/token
  FFN:        SwiGLU (3 matrices: gate+up+down), 3.0× expansion (6144 / 2048)
  Context:    40,960 tokens ≈ 31K words ≈ 82 pages (500 tok/page)
              RoPE base 1,000,000 — high base → designed for long-context extrapolation
  KV cache:   50 KB/token → 2.0 GB at full 40K context (bf16)
  QK-Norm:    Yes — per-head query/key normalization, stabilizes training at scale
  MoE:        Dense model — all 28 FFN layers are always active
```

For MoE models, the MoE line becomes:
```
  MoE:        128 experts, 8 active per token (6.25% utilization)
              Active params = 3.0B / Total params = 30.0B
```

---

## CLI Reference

```
usage: llm_stats.py [-h] [--format {table,json,csv}] [--token TOKEN]
                    [--revision REVISION] [--seq-len N] [--dump-raw]
                    [--timeout TIMEOUT]
                    models [models ...]

positional arguments:
  models                HuggingFace model IDs (e.g. Qwen/Qwen3-1.7B)

options:
  --format              Output format: table (default), json, csv
  --token               HuggingFace access token for gated models
                        (or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN env var)
  --revision            Git branch, tag, or commit SHA (default: main)
  --seq-len N           Sequence length for VRAM estimate
                        (default: model's max_position_embeddings)
  --dump-raw            Print raw config.json and exit — useful for debugging
  --timeout             HTTP request timeout in seconds (default: 30)
```

Progress messages go to **stderr**, so output can be cleanly redirected:
```bash
python llm_stats.py --format json Qwen/Qwen3-1.7B 2>/dev/null > stats.json
```

---

## What Can (and Cannot) Be Derived from config.json

### Derived automatically

| Stat | Source |
|---|---|
| Vocab size, context length, layer count, head counts | Direct fields |
| Head dimension | `head_dim` field, or computed as `hidden_size / num_attention_heads` |
| Attention type (MHA/MQA/GQA) | Ratio of `num_attention_heads` to `num_key_value_heads` |
| FFN expansion ratio | `intermediate_size / hidden_size` |
| RoPE base | `rope_theta` (or `rotary_emb_base` alias) |
| Total / active parameter count | Analytical formula — see below |
| KV cache bytes per token | `2 × kv_heads × head_dim × layers × 2 bytes` |
| VRAM estimates | Weights + KV cache + overhead |
| Tokenizer class | `tokenizer_class` in `tokenizer_config.json` |
| MoE detection | Presence of `num_experts` > 1 |

### Parameter count formula

```
embed         = vocab_size × hidden_size
lm_head       = 0  if tie_word_embeddings  else  vocab_size × hidden_size
per_attn_norm = hidden_size
per_attn      = Q(nh×dh×h) + K(nkv×dh×h) + V(nkv×dh×h) + O(nh×dh×h)
per_ffn_norm  = hidden_size
per_ffn       = gate(h×ffn) + up(h×ffn) + down(ffn×h)    ← SwiGLU (3 matrices)
              = up(h×ffn) + down(ffn×h)                   ← standard (2 matrices, e.g. GPT-2)
final_norm    = hidden_size

total = embed + lm_head + L × (per_attn_norm + per_attn + per_ffn_norm + per_ffn) + final_norm
active (MoE)  = same formula but with num_experts_per_tok experts instead of all experts
```

### Requires reading the paper

These stats are **not** in `config.json` and cannot be computed automatically:

- Training data (token count, composition, sources)
- Benchmark scores (MMLU, HumanEval, etc.)
- Inference throughput or latency (hardware-dependent)
- Pre-norm vs. post-norm placement (nearly all modern models use pre-norm)
- RoPE scaling semantics — `rope_scaling` dict exists but interpretation varies by paper (YaRN, LongRoPE, NTK-aware)
- Multi-Token Prediction (MTP) auxiliary heads
- QK-Norm for models that don't expose a `qk_norm` field (only Qwen3/Gemma3 do)
- Per-layer attention pattern (sliding window vs. full) — Gemma3 exposes `layer_types`; most models don't

---

## Supported Model Families

The script handles field name variations across model families automatically via `FIELD_ALIASES`. Most models work out of the box. A small set of model types have structural quirks that require special handling:

| Model type | Quirk |
|---|---|
| `gpt2`, `gpt_neo` | No `intermediate_size` field (derived as `4 × hidden_size`); 2-matrix FFN (no gate); absolute positional embeddings |
| `bloom` | ALiBi positional encoding (no RoPE theta) |
| `opt` | Absolute positional embeddings; `intermediate_size` may be absent |
| `phi`, `phi3` | Partial rotary factor |
| `gemma`, `gemma2`, `gemma3` | 4 norms per layer (vs. 2 in standard Llama-style models) |

### Handling a new or unknown model

If a model type is not recognized, the script still works generically — any fields it can find are filled in, missing ones show as `?`.

1. Run `--dump-raw` to inspect the raw `config.json` in your terminal
2. If a field is present but under a different name, add it to `FIELD_ALIASES` in `llm_stats.py`
3. If the model has a structural difference (e.g., non-RoPE positional encoding, absent `intermediate_size`, extra norms per layer), add a small `_xxx_overrides()` function and register it in `QUIRK_OVERRIDES`

The `missing_fields` list is always shown at the bottom of the output, listing any canonical fields that couldn't be resolved.

---

## VRAM Estimation Details

VRAM = **Weights + KV cache + 1 GB overhead** (framework, activations, buffers).

Weight memory at each quantization level:
```
bf16:  total_params × 2 bytes
int8:  total_params × 1 byte
int4:  total_params × 0.5 bytes
```

KV cache (always bf16):
```
per token = 2 × num_kv_heads × head_dim × num_layers × 2 bytes
            ↑ K and V       ↑ per head                 ↑ bf16
```

Use `--seq-len` to estimate VRAM at a realistic deployment sequence length instead of the model's full maximum context (which is often much larger than what you'd use in practice):

```bash
# Mistral-7B at 4K context vs. its full 32K context
python llm_stats.py --seq-len 4096 mistralai/Mistral-7B-v0.3
```

---
