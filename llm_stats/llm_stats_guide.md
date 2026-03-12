# Understanding LLM Architecture Stats

This guide explains every column and concept produced by `llm_stats.py` in simple language.

---

## How a Transformer LLM Is Built

Before diving into columns, here's the minimum mental model you need.

A language model takes a sequence of text, converts it to numbers, and predicts what comes next. Internally, it's a stack of identical **layers** (also called blocks or transformer blocks). Each layer has two parts:

1. **Attention** - tokens look at each other and decide what context is relevant
2. **FFN (Feed-Forward Network)** - each token independently processes information through a small neural network

Information flows through all layers in order, from bottom to top. At the end, a final linear projection converts the internal representation back into vocabulary probabilities.

Every number stored in the model is called a **parameter** (also called a weight). More parameters = more capacity to store knowledge, but also more memory and compute required.

---

## Section 1 - Architecture Table

### `Model` and `Type`

`Model` is the HuggingFace identifier (`org/model-name`). `Type` is the `model_type` field from the model's `config.json` - it identifies the model family and determines which code is used to run it. Examples: `llama`, `qwen3`, `mistral`, `gpt2`, `gemma3`, `qwen3_moe`.

Two models can have the same `Type` but different sizes (e.g. Llama-3.2-1B and Llama-3.1-70B are both `llama`).

---

### `Tokenizer`

Before text enters the model, it must be broken into **tokens** - the atomic units the model operates on. A tokenizer is the component that does this conversion.

A single word can become one token (`"cat"`) or several (`"unbelievable"` → `["un", "believ", "able"]`). Common words tend to be single tokens; rare words or code may split into many.

The `Tokenizer` column shows the `tokenizer_class` field from `tokenizer_config.json`. Examples:

| Tokenizer class | Used by |
|---|---|
| `Qwen2Tokenizer` | Qwen2, Qwen3 |
| `LlamaTokenizer` | Llama, Mistral |
| `BloomTokenizerFast` | BLOOM |
| `PreTrainedTokenizerFast` | many models |

**Why it matters:** Two models with the same tokenizer share a vocabulary and can often be compared directly on token counts. Models with different tokenizers may tokenize the same text differently, making it hard to compare prompt lengths directly.

---

### `Vocab`

The number of unique tokens the model knows about. Think of it as the model's "alphabet" - but at the word-piece level rather than character level.

| Model | Vocab |
|---|---|
| GPT-2 | 50.3K |
| Mistral-7B | 32.8K |
| Qwen3 | 151.9K |
| BLOOM | 250.9K |

**Interpretation:**
- Larger vocab → each common concept maps to fewer tokens → shorter sequences → faster inference and lower KV cache usage.
- Larger vocab also means a larger **embedding table** (one vector per token), which increases parameter count. Qwen3's 151K vocab contributes ~18% of its 1.7B total parameters.
- A very small vocab (like early GPT-2's 50K) means more tokens per concept, which wastes context space.

---

### `Context`

The maximum number of tokens the model can process in a single forward pass - effectively the model's "working memory window." This is the `max_position_embeddings` field.

| Model | Context |
|---|---|
| GPT-2 | 1K |
| OPT-6.7B | 2K |
| Mistral-7B | 32K |
| Qwen3 | 40K |
| Llama-3.1-8B | 128K |

**Humanized:** The script converts this to words (÷ 1.3 tokens/word) and pages (÷ 500 tokens/page) so you can intuitively grasp it.

**Interpretation:**
- Context window limits how much text the model can "see" at once. A 32K context can hold roughly one long novel chapter or a medium-sized codebase.
- Larger context = more KV cache memory (the cache grows linearly with sequence length - see [KV-Cache/tok](#kv-cachetok) below).
- Context length in `config.json` is the *maximum trained* length. Using the full context is often impractical due to VRAM cost, which is why `--seq-len` exists.

---

### `Layers`

The number of transformer blocks stacked on top of each other. Each layer adds depth - the model processes the same sequence again with different learned weights.

| Model size | Typical layers |
|---|---|
| ~100M (GPT-2) | 12 |
| ~1B | 16–24 |
| ~7B | 32 |
| ~30B | 48 |
| ~70B | 80 |

**Interpretation:**
- More layers = deeper reasoning capability. A 32-layer model can perform 32 rounds of "look at the context and update my understanding" per token.
- Depth and width (hidden size) both contribute to capacity. Doubling layers doubles total parameters; doubling hidden size quadruples them (because matrices scale as `hidden × hidden`).
- The **Structure** line in Model Insights (`N layers × M params/layer = X total layer params`) shows how parameters concentrate per layer.

---

### `Heads`

The number of **query heads** in the multi-head attention mechanism.

**What attention does:** Each token generates a query ("what am I looking for?"), and all tokens generate keys ("what do I offer?") and values ("what information do I carry?"). The query compares against all keys to decide how much to attend to each token's value.

**What heads do:** Instead of one big attention computation, the model runs several smaller, parallel attention computations simultaneously - each "head" looks at the context from a different learned perspective. Some heads might specialize in syntax; others in long-range references; others in local word order.

The output of all heads is concatenated and projected back to the hidden dimension.

**Interpretation:**
- More heads = richer representational diversity, but each head becomes narrower (head_dim = hidden_size / num_heads).
- Typical values: 8–96 heads depending on model size.
- The number of query heads does **not** have to equal the number of KV heads (see `KV-Heads` below).

---

### `KV-Heads`

The number of **key/value heads** - which may be fewer than the query heads.

This is the crux of a major efficiency innovation called **Grouped Query Attention (GQA)** and its extreme variant **Multi-Query Attention (MQA)**. To understand it, you first need to understand the **KV cache**.

#### The KV Cache Problem

During text generation, the model produces one token at a time. When generating token #100, it needs to attend to tokens #1 through #99. Computing keys and values for all prior tokens from scratch every step would be extremely slow. So the model **caches** the key and value vectors for every prior token in memory - this is the KV cache.

The KV cache grows with sequence length and model size. For a long conversation or document, it can consume as much memory as the model weights themselves.

#### MHA vs GQA vs MQA

| Name | How it works | KV-Heads |
|---|---|---|
| **MHA** (Multi-Head Attention) | Every query head has its own K and V head | = Heads |
| **GQA** (Grouped Query Attention) | N query heads share one K/V head | Heads ÷ N |
| **MQA** (Multi-Query Attention) | All query heads share a single K/V head | 1 |

The `Attn-Type` column tells you which variant applies and the reduction factor (e.g. `GQA-4x` means 4 query heads per KV head).

**Real examples:**

```
GPT-2:        Heads=12, KV-Heads=12  → MHA     (full KV cache)
Mistral-7B:   Heads=32, KV-Heads=8   → GQA-4x  (KV cache 4× smaller than MHA)
Qwen3-1.7B:   Heads=16, KV-Heads=8   → GQA-2x  (KV cache 2× smaller)
Qwen3-30B-A3B: Heads=32, KV-Heads=4  → GQA-8x  (KV cache 8× smaller)
```

**Interpretation:**
- GQA reduces KV cache size proportionally: `GQA-4x` uses 4× less KV memory than MHA with the same hidden size.
- This is why modern large models almost universally use GQA - it lets them serve longer contexts without running out of memory.
- Empirically, GQA quality is close to MHA quality; MQA (all heads share 1 KV) degrades quality more noticeably.

---

### `HeadDim`

The dimension of each individual attention head's vector space. It's usually computed as:

```
head_dim = hidden_size / num_attention_heads
```

Some models (e.g. Qwen3) specify it explicitly in the config.

**Real examples:**
- GPT-2 (hidden=768, heads=12): `head_dim = 768 / 12 = 64`
- Mistral-7B (hidden=4096, heads=32): `head_dim = 4096 / 32 = 128`
- Qwen3-1.7B (hidden=2048, heads=16): `head_dim = 128` (explicit)

**Interpretation:**
- `head_dim = 64` is a GPT-2-era default. Most modern models use `128` or `256`.
- Larger head_dim means each head operates in a richer vector space.
- head_dim directly determines KV cache cost per head: more head_dim = more bytes cached per token per head (see [KV-Cache/tok](#kv-cachetok)).
- When head_dim doesn't evenly divide from `hidden_size / heads`, the config usually has an explicit `head_dim` field.

---

### `FFN-Size`

The size of the **hidden layer inside the Feed-Forward Network** (FFN) block - also called `intermediate_size` in most configs.

Each transformer layer has an FFN that expands the token's hidden representation into a larger space, applies a non-linearity, and projects back down:

```
FFN:  hidden → intermediate → hidden
      [2048]  →   [6144]    → [2048]    (Qwen3-1.7B example)
```

This expansion-and-contraction is where much of the model's "knowledge storage" happens. The FFN parameters vastly outnumber the attention parameters in most models.

**Real examples:**
- GPT-2: 768 → 3072 → 768 (2-matrix, no gate)
- Qwen3-1.7B: 2048 → 6144 → 2048
- Mistral-7B: 4096 → 14336 → 4096

**Interpretation:**
- The FFN is generally the largest component (see [FFN%](#parameter-breakdown-table)).
- A larger `FFN-Size` relative to `hidden_size` = more capacity to memorize patterns.
- MoE models use a much smaller per-expert FFN, since only a few experts activate per token.

---

### `FFN-Ratio`

`FFN-Size / hidden_size` - the expansion factor of the FFN.

**Real examples:**
- GPT-2: 4.0× (`3072 / 768`)
- Mistral-7B: 3.5× (`14336 / 4096`)
- Qwen3 (all sizes): 3.0× (`6144 / 2048`, `12288 / 4096`)
- OPT-6.7B: 4.0× (`16384 / 4096`)

**Interpretation:**
- The classical default is `4×`, from the original "Attention is All You Need" paper.
- Modern models have moved toward `~3×` (especially when using SwiGLU, which adds a third matrix - the gate). The total parameter count is similar because SwiGLU adds parameters even at a lower expansion ratio.
- A very high ratio (>5×) bets heavily on FFN-driven capacity. A low ratio (<2.5×) saves parameters there and spends them elsewhere (e.g. more layers or larger hidden size).

---

### `Attn-Type`

The attention variant: `MHA`, `MQA`, or `GQA-Nx`. This is derived from the ratio of query heads to KV heads.

See [KV-Heads](#kv-heads) above for a full explanation.

---

### `Activation`

The non-linear function applied inside the FFN. Without non-linearity, stacking linear layers would collapse to a single linear transformation.

| Activation | Used by | Notes |
|---|---|---|
| `gelu` / `gelu_new` | GPT-2, BERT | Smooth approximation; classic |
| `relu` | OPT, early models | Simple, fast, but "dead neuron" problem |
| `silu` | Llama, Qwen3, Mistral | Also called Swish; smooth, works well with SwiGLU |
| `gelu_pytorch_tanh` | GPT-4 style, Gemma | Tanh approximation of GELU |

**SwiGLU**: Many modern models use a **gated** FFN called SwiGLU, where the activation is `silu` but there's an extra gate matrix that controls information flow:

```
output = (gate_proj(x) * sigmoid(gate_proj(x))) × up_proj(x)
output = down_proj(output)
```

This requires **3 projection matrices** (gate, up, down) instead of the classical 2 (up, down), hence `SwiGLU - 3 matrices (gate+up+down)` in Model Insights.

**Why it matters:** SwiGLU consistently outperforms plain GELU/ReLU in practice. The extra matrix cost is worth it - modern models absorb this by slightly reducing the expansion ratio (3× instead of 4×).

---

### `QK-Norm`

Whether **per-head query/key normalization** is applied before computing attention scores.

In standard attention, raw query and key vectors are dot-producted together. As models get very deep or very large, these dot products can become extremely large in magnitude, causing gradient instability during training. QK-Norm adds a normalization step (RMSNorm) to each head's query and key vectors before the dot product, keeping magnitudes bounded.

**Current status:**
- `Yes`: Qwen3, Gemma3 - explicitly expose `qk_norm: true` in their configs
- `?`: Most models don't expose this field. It may be absent (false) or simply undocumented in `config.json`

**Interpretation:**
- QK-Norm is a training stability technique that doesn't change what the model can represent, only how stably it trains. Models that expose `qk_norm: true` are likely trained at larger scale where instability becomes a real concern.

---

### `Pos-Enc`

How the model encodes **position** - i.e., how it knows that token #5 comes before token #6. Transformers have no built-in sense of order; position must be injected.

| Pos-Enc | Full name | How it works |
|---|---|---|
| **RoPE** | Rotary Position Embedding | Rotates query/key vectors by an angle proportional to their position. The rotation naturally encodes relative distances. Used by almost all modern models (Llama, Qwen, Mistral, Gemma). |
| **APE** | Absolute Positional Embedding | Adds a learned or fixed vector for each absolute position index (position 0, 1, 2, ...) to the token embeddings. Used by GPT-2, OPT, early BERT. Has a hard maximum at the training length. |
| **ALiBi** | Attention with Linear Biases | Adds a position-dependent bias to attention scores - closer tokens get less penalty, farther tokens get more. Used by BLOOM. No positional embeddings at all. |

**Interpretation:**
- **RoPE** generalizes better to lengths beyond training. The math encodes relative distance directly, not absolute position.
- **APE** models have a strict context ceiling - you cannot extrapolate beyond `max_position_embeddings` without retraining or fine-tuning.
- **ALiBi** is simple and extrapolates reasonably well, but is less commonly used in frontier models today.

---

### `RoPE-Base`

The `rope_theta` parameter - a scalar that controls how quickly the rotation angles decay with distance in RoPE. Only relevant for RoPE models (`N/A` shown for APE/ALiBi).

```
angle for position p, dimension d = p / (rope_theta ^ (2d / head_dim))
```

A higher `rope_theta` means each dimension's angle rotates more slowly across positions, making the encoding more sensitive to long-range differences.

| RoPE-Base | Implies |
|---|---|
| 10,000 | Original Llama 1 / GPT-NeoX default - short-to-medium context |
| 500,000 | Llama 3.2 - explicitly tuned for long context |
| 1,000,000 | Qwen3, Mistral-7B v0.3 - very long context |
| 10,000,000+ | Some research models targeting 1M+ token contexts |

**Interpretation:**
- Very high RoPE base (≥500K) is a signal that the model was **trained on long contexts** and designed to extrapolate to even longer ones.
- A model with `rope_theta=10000` and `max_position_embeddings=4096` is a shorter-context model - don't assume it handles 32K just by changing settings.
- You cannot simply set a high `rope_theta` at inference time on a model trained with a low one and expect good behavior - the base must match training.

---

### `Dtype`

The numeric precision of the stored model weights.

| Dtype | Bytes/param | Notes |
|---|---|---|
| `float32` | 4 | Full precision; used during training |
| `float16` | 2 | Half precision; older standard for inference |
| `bfloat16` | 2 | Brain float; same memory as float16, better range; preferred for modern models |
| `int8` | 1 | Quantized; significant compression |
| `int4` | 0.5 | Aggressively quantized; runs on smaller GPUs |

**What `torch_dtype` in config.json means:** This is the dtype the model was *released in* - the native precision. The VRAM estimate for "weights-bf16" uses this to compute a reference weight size (treating the config dtype as if it were bf16).

**Interpretation:**
- Most modern models are released in `bfloat16`. The `?` you see in the GPT-2 row means `torch_dtype` was absent from its config - it's an older model that predates this field.
- Quantized formats (int8, int4) reduce weight size at the cost of some precision. The VRAM table shows weight sizes at each level, while keeping the KV cache at bf16 (quantizing KV cache requires special inference software and is rarely done).

---

### `KV-Cache/tok`

How many bytes the KV cache grows by for **each additional token** in the sequence. This is a derived value, not in `config.json`:

```
KV-Cache/tok = 2 × num_kv_heads × head_dim × num_layers × dtype_bytes
               ↑ K and V       ↑ per head              ↑ bf16 = 2 bytes
```

The factor of 2 accounts for both the key cache and the value cache. Every token that enters the context window requires storing both K and V vectors across all layers and all KV heads.

**Real examples:**

```
GPT-2 (MHA):      2 × 12 × 64  × 12 × 2 =  36 KB/token
OPT-6.7B (MHA):   2 × 32 × 128 × 32 × 2 = 512 KB/token   ← large MHA cost
Qwen3-1.7B (GQA): 2 × 8  × 128 × 28 × 2 = 112 KB/token
Mistral-7B (GQA): 2 × 8  × 128 × 32 × 2 = 128 KB/token
Qwen3-8B (GQA):   2 × 8  × 128 × 36 × 2 = 144 KB/token
Qwen3-30B (GQA):  2 × 4  × 128 × 48 × 2 =  96 KB/token   ← GQA-8x keeps this low
```

Notice how GQA dramatically reduces per-token cost. The 30B model has fewer bytes/token than the 1.7B model because it uses `GQA-8x` (4 KV heads) vs `GQA-2x` (8 KV heads).

**Why this matters:** Multiply `KV-Cache/tok` by the number of tokens in your sequence to get the total KV cache cost:
```
Total KV = KV-Cache/tok × sequence_length
```
At 40K tokens, Qwen3-1.7B needs `112 KB × 40,960 ≈ 4.4 GB` just for the KV cache - more than the model weights themselves at int4.

---

## Section 2 - Parameter Breakdown Table

### `Total` and `Active`

`Total` is the full parameter count. `Active` is how many parameters are actually engaged in one forward pass.

For **dense models**, every parameter is used for every token → Active = Total → shown as `-` (same value, no distinction needed).

For **MoE (Mixture of Experts) models**, the FFN layer contains many "expert" sub-networks, but only a few activate per token. The router picks the top-K experts:

```
Qwen3-30B-A3B: Total=30.5B, Active=3.4B (only 8 of 128 experts activate per token)
```

This is the core MoE tradeoff: you get a large total knowledge base (30.5B parameters worth of patterns) but spend compute as if running a ~3.4B model. The cost of loading all weights into VRAM remains, though - you need VRAM for the full 30.5B.

---

### `Embed (%)` and `LM-Head (%)`

**Embedding table** (`Embed`): The first thing the model does is look up a learned vector for each input token. This lookup table has shape `vocab_size × hidden_size`.

```
Qwen3-1.7B embed params: 151,936 × 2,048 = 311M   (18% of 1.7B total)
Mistral-7B embed params:  32,768 × 4,096 = 134M   (2% of 7.2B total)
```

A larger vocab or larger hidden size → larger embedding table. This is why Qwen3's 152K vocab makes the embedding take up a full 18% of the model, while Mistral's 33K vocab takes only 2%.

**LM Head** (`LM-Head`): At the end of the model, a linear projection maps the final hidden state back to vocabulary probabilities. Its shape is `hidden_size × vocab_size` - the same shape as the embedding table transposed.

Many models **tie** these weights (`tie_word_embeddings: True`), meaning the LM head is literally the embedding table in reverse, stored once. In that case:
- `LM-Head% = 0%` - no extra parameters
- These models show `0% (shared weights, no extra params)` in Model Insights

When weights are **not tied**, the LM head is a separate parameter matrix:
```
Mistral-7B:  32,768 × 4,096 = 134M LM-head params → 2% of total
GPT-2:       50,257 × 768   =  38M LM-head params → 24% of total (large fraction for a small model)
```

**Interpretation:** In small models, the embedding + LM-head together can eat a surprising fraction of the budget. GPT-2's embedding + LM-head = 48% of all 162M parameters. This is why smaller models often tie weights - it's free to share.

---

### `Attn (%)` - Attention Parameter Share

All attention projection matrices across all layers combined.

**Per attention layer:**
```
Q projection: hidden × (num_heads × head_dim)
K projection: hidden × (num_kv_heads × head_dim)
V projection: hidden × (num_kv_heads × head_dim)
O projection: (num_heads × head_dim) × hidden
+ layer norm before attention
```

With GQA, K and V are smaller (fewer KV heads), so the attention parameter fraction shrinks:

```
GPT-2 (MHA):      Attn = 17%
OPT-6.7B (MHA):   Attn = 31%   ← 31% with full MHA on a medium model
Qwen3-1.7B (GQA): Attn = 20%   ← GQA means smaller K/V matrices
Mistral-7B (GQA): Attn = 19%
Qwen3-30B (GQA):  Attn =  3%   ← MoE model: FFN dominates even more
```

---

### `FFN (%)` - FFN Parameter Share

All FFN projection matrices across all layers combined. Almost always the dominant component.

```
GPT-2:         FFN = 35%
OPT-6.7B:      FFN = 63%
Qwen3-1.7B:    FFN = 61%
Mistral-7B:    FFN = 78%
Qwen3-30B-A3B: FFN = 95%   ← MoE: essentially all parameters are expert weights
```

**Why FFN dominates:** Each FFN layer has matrices sized `hidden × intermediate` (plus `intermediate × hidden`), which scale as the product of two large dimensions. At `4×` expansion and 32 layers, the FFN contains `2 × 4 × hidden² × 32` parameters - quadratic in `hidden_size`.

**The MoE extreme:** In `Qwen3-30B-A3B`, 95% of all parameters are inside expert FFNs. The model is basically "a small attention mechanism routing through a very large bank of specialist FFNs." Because most experts are idle per token, the effective compute stays low, but all 30B parameters must reside in VRAM.

**Interpretation guide:**

| FFN% | What it suggests |
|---|---|
| 30–40% | Small model or heavy attention investment |
| 55–70% | Standard dense model (typical range) |
| 70–85% | Large dense model with wide FFNs |
| 90–95% | MoE model (nearly all params are expert weights) |

---

## Section 3 - VRAM Estimates

### Weights at Different Quantization Levels

**`Weights-bf16`**: The full-precision weight memory. `total_params × 2 bytes`.

**`Weights-int8`**: Weights quantized to 8-bit integers. `total_params × 1 byte`. Half the memory of bf16.

**`Weights-int4`**: Weights quantized to 4-bit integers. `total_params × 0.5 bytes`. One quarter of bf16. The minimum typically used in practice (going lower degrades quality significantly).

**What quantization does:** Rather than storing each weight as a 16-bit float (65,536 possible values), quantization maps weights to 8-bit (256 values) or 4-bit (16 values) integers using a learned or block-wise scaling factor. This is lossy compression - some precision is lost, but in practice the quality degradation is small at int8 and acceptable at int4 for many tasks.

**The tradeoff:** Quantization is for inference only (training requires higher precision). You're trading some accuracy for the ability to run on a smaller GPU.

---

### `KV@MaxCtx` and `KV@SeqLen`

`KV@MaxCtx` is the KV cache size at the model's full `max_position_embeddings`:
```
KV@MaxCtx = KV-Cache/tok × max_position_embeddings
```

`KV@SeqLen` is the KV cache at the sequence length you specify with `--seq-len`. If you don't pass `--seq-len`, it equals `KV@MaxCtx`.

**Why this matters enormously:**

```
Qwen3-1.7B at full 40K context:  KV = 112 KB × 40,960 = 4.4 GB
Qwen3-1.7B at 4K context:        KV = 112 KB × 4,096  = 0.4 GB
                                           ↑ 10× less memory at 4K vs 40K
```

The KV cache scales linearly with sequence length. For short conversations you rarely need the full context, and using `--seq-len` with a realistic value gives a much more accurate VRAM estimate for your actual use case.

**Key insight:** For large models, the KV cache can dominate over model weights at long context lengths. Mistral-7B at its full 32K context:
```
Weights (bf16): 13.5 GB
KV@MaxCtx:       4.0 GB
Total:          18.5 GB
```
Quantizing the weights to int4 saves ~10GB on weights, but the KV cache stays at 4GB (still bf16). This is why quantization doesn't make running long-context models free - the KV cache is a separate fixed cost.

---

### `Total-bf16 / int8 / int4`

Full VRAM requirement at each quantization level:
```
Total = Weights (at quantization level) + KV Cache (always bf16) + 1 GB overhead
```

The **1 GB overhead** accounts for framework memory (PyTorch, CUDA context), intermediate activations, and other buffers. It's a conservative estimate; very large models may need more overhead.

**Real comparison:**
```
Mistral-7B:
  bf16:  13.5 GB weights + 4.0 GB KV + 1 GB = 18.5 GB  (needs A100 or RTX 3090)
  int8:   6.8 GB weights + 4.0 GB KV + 1 GB = 11.8 GB  (needs RTX 3080 Ti / 4080)
  int4:   3.4 GB weights + 4.0 GB KV + 1 GB =  8.4 GB  (needs RTX 3080 12GB or better)
```

Notice: quantizing from bf16 to int4 saves ~10 GB on weights, but the 4 GB KV cache and 1 GB overhead are fixed. You can't quantize your way out of KV cache costs.

---

### `Min-GPU (int4)`

The smallest standard GPU tier whose VRAM fits the `Total-int4` figure. The tiers used:

| Tier | Representative GPUs |
|---|---|
| 4 GB | Entry-level consumer |
| 8 GB | RTX 3070, RTX 4060 |
| 12 GB | RTX 3080, RTX 4070 |
| 16 GB | RTX 3080 Ti, RTX 4080 |
| 24 GB | RTX 3090, RTX 4090, A10 |
| 40 GB | A100 40GB |
| 48 GB | A6000, L40S |
| 80 GB | A100 80GB, H100 |
| 141 GB | H200 |

**Interpretation:** This is the floor, not the recommendation. At "minimum," you're running at maximum context with no headroom. In practice, you want 20–30% spare VRAM for safety, especially when running longer contexts or batching multiple requests.

```
Qwen3-1.7B int4 total = 6.2 GB → Min-GPU: 8 GB  (RTX 3070/4060)
Mistral-7B  int4 total = 8.4 GB → Min-GPU: 12 GB (RTX 3080/4070)
Qwen3-30B   int4 total = 19 GB  → Min-GPU: 24 GB (RTX 3090/4090)
```

If the `>141GB` tier appears, it means the model doesn't fit on a single H200 at int4 - it requires multi-GPU or CPU offloading even in its most compressed form.

---

## Section 4 - Model Insights (Narrative)

### `Structure` line

```
28 layers × 50.3M params/layer = 1.4B total layer params
```

Shows how the parameter budget distributes per layer. This is useful for comparing architectures of similar total size: a model with 48 thin layers vs 24 wide layers will show different params/layer.

---

### `Budget` line

```
Embed 18% | Attn 20% | FFN 61% | LM-head 0% (shared weights, no extra params)
```

The parameter budget allocation. See the [Embed%](#embed--and-lm-head-) and [FFN%](#ffn---ffn-parameter-share) sections above. This line is the quickest way to see a model's architectural philosophy at a glance.

---

### `Attention` line

```
GQA-4x - 32 Q-heads share 8 KV-heads; KV cache is 4× smaller than MHA
head_dim=128; each layer caches 2,048 values/token (K + V combined)
```

The `2,048 values/token` is computed as `(K heads + V heads) × head_dim = (8 + 8) × 128 = 2,048`. Over 32 layers, one new token adds `2,048 × 32 = 65,536` values to the KV cache.

---

### `FFN` line

```
SwiGLU - 3 matrices (gate+up+down), 3.5× expansion (14,336 / 4,096)
```

or for an older model:

```
RELU - 2 matrices (up+down), 4.0× expansion (16,384 / 4,096)
```

This tells you both the FFN variant and the expansion ratio. The number of matrices matters for understanding parameter counts and compute cost per token.

---

### `MoE` line

**Dense:**
```
Dense - all 32 FFN layers are always active (100% compute utilization)
```

**MoE:**
```
128 total experts, 8 active per token (6.2% compute utilization per layer)
Active = 3.4B / Total = 30.5B - 11% of weights engaged per forward pass
Per-expert FFN: 768 intermediate dim (0.38× expansion, smaller than dense FFN by design)
```

The "per-expert FFN" note is important: MoE models use a much smaller intermediate size per expert (e.g. 768) compared to dense FFNs (6144 for the same hidden size). The total capacity comes from having many experts, not from each expert being large.

---

### `Context` line

```
40,960 tokens ≈ 31,508 words ≈ 82 pages (at ~500 tok/page)
RoPE base 1,000,000 - very high RoPE base → explicitly designed for long-context extrapolation
```

The humanized context length and the RoPE base signal. See [Context](#context) and [RoPE-Base](#rope-base) above for interpretation.

---

### `KV cache` line

```
112 KB/token → 4.4 GB at full 40K context (bfloat16)
```

The per-token KV cost (from the [KV-Cache/tok formula](#kv-cachetok)) scaled to the full context window. This is the single most useful number for understanding why long-context inference is expensive.

---

### `VRAM` lines

```
~8.6GB total = 3.2GB weights + 4.4GB KV + 1GB overhead
Minimum GPU to run at int4: 8GB - RTX 3070/4060 (8GB)
int4 total = 6.2GB (0.8GB weights + 4.4GB KV + 1GB overhead)
```

The narrative breakdown of the VRAM table, showing the three additive components explicitly. Especially useful for understanding why a quantized model still requires significant VRAM - often because the KV cache dominates.

---

## Common Patterns to Look For When Comparing Models

**Does GQA matter?** Compare `OPT-6.7B (MHA, KV=512KB/tok)` vs `Mistral-7B (GQA-4x, KV=128KB/tok)`. The 7B Mistral uses 4× less KV memory per token despite being a larger model, purely from GQA.

**Does MoE make sense?** `Qwen3-30B-A3B (Active=3.4B, Total=30.5B)` runs with the compute cost of a ~3B model but carries knowledge from a 30B parameter bank. The tradeoff: you need 24GB VRAM to hold all 30.5B parameters at int4, even though you only compute with 3.4B per token.

**Tied vs untied embeddings in small models:** GPT-2 (162M) spends 48% of its parameters on the embedding + LM-head. Tying them would cut total params by ~24% for free. This is why modern small models almost always tie embeddings.

**RoPE base as a long-context signal:** A model with `RoPE-Base = 10,000` at `Context = 4K` is a short-context model. A model with `RoPE-Base = 1,000,000` at `Context = 40K` is explicitly designed for long-context - the high base lets the model maintain coherent position information over long distances.

**Why FFN% grows with model size:** As models get larger, the hidden size grows (say, 2048 → 4096 → 8192). Embedding tables grow linearly with `hidden_size`. But FFN parameters grow as `~hidden_size²` (because the matrices are `hidden × intermediate`). So in larger models, FFN naturally eats a larger share of the budget.

---

## Appendix - Data Provenance: Where Every Column Comes From

The script fetches exactly two lightweight JSON files per model - no weights downloaded:

| File | URL pattern | Size |
|---|---|---|
| `config.json` | `https://huggingface.co/{model}/resolve/main/config.json` | ~5–50 KB |
| `tokenizer_config.json` | `https://huggingface.co/{model}/resolve/main/tokenizer_config.json` | ~1–5 KB |

Everything below is either read directly from one of these files, or mathematically derived from values in them. Nothing requires the model weights.

---

### Architecture Table

| Column | Source | Field(s) tried (in priority order) | Notes |
|---|---|---|---|
| **Model** | CLI argument | - | Last 1–2 path components of the HuggingFace model ID you pass |
| **Type** | `config.json` | `model_type` | Identifies the model family; drives quirk overrides |
| **Tokenizer** | `tokenizer_config.json` | `tokenizer_class` | Absent → `?` |
| **Vocab** | `config.json` | `vocab_size` | Absent → `?` |
| **Context** | `config.json` | `max_position_embeddings`, `n_positions`, `max_seq_len`, `max_sequence_length`, `n_ctx` | First match wins |
| **Layers** | `config.json` | `num_hidden_layers`, `n_layer`, `n_layers`, `num_layers` | First match wins |
| **Heads** | `config.json` | `num_attention_heads`, `n_head`, `n_heads`, `num_heads` | Query head count |
| **KV-Heads** | `config.json` | `num_key_value_heads`, `num_kv_heads`, `n_kv_heads` | If absent → defaults to `Heads` (MHA assumed) |
| **HeadDim** | `config.json` → derived | `head_dim` (explicit). If absent: `hidden_size ÷ num_attention_heads` | Some models (Qwen3) set this explicitly |
| **FFN-Size** | `config.json` → derived | `intermediate_size`, `ffn_dim`, `inner_dim`. GPT-2/OPT quirk: derived as `4 × hidden_size` when absent | MoE models: per-expert size from `moe_intermediate_size` |
| **FFN-Ratio** | Derived | - | `intermediate_size ÷ hidden_size` |
| **Attn-Type** | Derived | - | `Heads == KV-Heads` → MHA; `KV-Heads == 1` → MQA; else GQA-{Heads÷KV-Heads}x |
| **Activation** | `config.json` | `hidden_act`, `activation_function`, `hidden_activation` | |
| **QK-Norm** | `config.json` | `qk_norm` | Most models don't expose this → `?` |
| **Pos-Enc** | Derived via quirks | - | GPT-2/OPT quirk → APE; BLOOM quirk → ALiBi; `rope_theta` present → RoPE; else `?` |
| **RoPE-Base** | `config.json` | `rope_theta`, `rotary_emb_base`, `rope_emb_base` | N/A for APE/ALiBi models |
| **Dtype** | `config.json` | `torch_dtype` | Absent in older models → `?` |
| **KV-Cache/tok** | Derived | - | `2 × KV-Heads × HeadDim × Layers × dtype_bytes` |

---

### Parameter Breakdown Table

All parameter counts are **analytically computed** from config fields. No weights are downloaded.

| Column | Formula | Config fields used |
|---|---|---|
| **Embed** | `vocab_size × hidden_size` | `vocab_size`, `hidden_size` |
| **Attn (all layers)** | `Layers × per_attn_layer` where `per_attn = Q + K + V + O + pre-norm + (QK-norm if present)` | `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `hidden_size`, `attention_bias`, `qk_norm` |
| - Q projection | `hidden_size × (num_heads × head_dim)` + bias if `attention_bias` | |
| - K projection | `hidden_size × (num_kv_heads × head_dim)` + bias if `attention_bias` | |
| - V projection | `hidden_size × (num_kv_heads × head_dim)` + bias if `attention_bias` | |
| - O projection | `(num_heads × head_dim) × hidden_size` + bias if `attention_bias` | |
| - Pre-attn norm | `hidden_size` | |
| - QK-Norm | `(num_heads + num_kv_heads) × head_dim` (only if `qk_norm=True`) | |
| **FFN (all layers)** | `Layers × per_ffn_layer` | `num_hidden_layers`, `hidden_size`, `intermediate_size`, `hidden_act` |
| - SwiGLU (silu/swish) | `(2 × hidden × ffn) + (ffn × hidden)` + pre-FFN norm | `intermediate_size` |
| - Standard 2-matrix (relu/gelu) | `(hidden × ffn) + (ffn × hidden)` + pre-FFN norm | `intermediate_size` |
| - MoE router | `hidden_size × num_experts` | `num_experts` |
| - MoE per-expert | `num_experts × 3 × hidden_size × moe_intermediate_size` | `num_experts`, `moe_intermediate_size` |
| - MoE shared experts | `num_shared_experts × 3 × hidden_size × shared_expert_intermediate_size` | `num_shared_experts`, `shared_expert_intermediate_size` |
| - Gemma3 extra norms | `+2 × hidden_size` per layer (quirk override) | `model_type` = gemma/gemma2/gemma3 |
| **LM-Head** | `vocab_size × hidden_size` if `tie_word_embeddings=False`, else `0` | `tie_word_embeddings`, `vocab_size`, `hidden_size` |
| **Total** | `Embed + Attn + FFN + LM-Head + final_norm` | all above |
| **Active** | Same as Total for dense models. For MoE: replaces all-expert FFN with `num_experts_per_tok` active experts | `num_experts_per_tok` |
| **Embed%** | `Embed ÷ Total` | - |
| **Attn%** | `Attn ÷ Total` | - |
| **FFN%** | `FFN ÷ Total` | - |
| **LM-Head%** | `LM-Head ÷ Total` | - |

---

### VRAM Table

| Column | Formula | Inputs |
|---|---|---|
| **Weights-bf16** | `total_params × 2 bytes` | `total_params` |
| **Weights-int8** | `total_params × 1 byte` | `total_params` |
| **Weights-int4** | `total_params × 0.5 bytes` | `total_params` |
| **KV@MaxCtx** | `kv_cache_bytes_per_token × max_position_embeddings` | derived KV/tok, `max_position_embeddings` |
| **Total-bf16** | `Weights-bf16 + KV + 1 GB overhead` | - |
| **Total-int8** | `Weights-int8 + KV + 1 GB overhead` | KV cache stays bf16 regardless of weight quantization |
| **Total-int4** | `Weights-int4 + KV + 1 GB overhead` | KV cache stays bf16 regardless of weight quantization |
| **Min-GPU (int4)** | Smallest tier ≥ Total-int4 from `[4, 8, 12, 16, 24, 40, 48, 80, 141]` GB | `total_int4` |

The **1 GB overhead** is a hardcoded constant for PyTorch/CUDA runtime, intermediate activations, and buffers.

---

### Fields That Cannot Be Derived from config.json

These are intentionally absent from the script's output:

| Stat | Why it's missing |
|---|---|
| Training data (tokens, sources) | Not in any config file; requires reading the model card or paper |
| Benchmark scores (MMLU, HumanEval, etc.) | Empirical; not in config |
| Inference throughput / latency | Hardware- and framework-dependent; not computable from architecture alone |
| Pre-norm vs post-norm placement | Almost always pre-norm in modern models, but not explicitly flagged in config |
| RoPE scaling type (YaRN, LongRoPE, NTK) | The `rope_scaling` field exists in some configs but the semantics vary by model |
| QK-Norm for models that don't expose it | Only Qwen3 and Gemma3 explicitly set `qk_norm: true`; others are unknown |
| Per-layer sliding window pattern | `sliding_window` in config is the window size, not which layers use it |
| Multi-Token Prediction (MTP) heads | Auxiliary training heads; not in standard config fields |
