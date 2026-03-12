#!/usr/bin/env python3
"""
LLM Architecture Stats Tool

Usage
-----
    python llm_stats.py Qwen/Qwen3-1.7B meta-llama/Llama-3.2-1B
    python llm_stats.py --seq-len 4096 mistralai/Mistral-7B-v0.3
    python llm_stats.py --format json Qwen/Qwen3-1.7B > stats.json
    python llm_stats.py --dump-raw new-org/new-model   # inspect raw config.json
    python llm_stats.py --token $HF_TOKEN gated/model

Dependencies
------------
    pip install requests   (only external dependency; no torch required)

"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps canonical field names → list of HF config.json keys tried in priority order.
# Add new aliases here when encountering a model with non-standard field names.
FIELD_ALIASES: dict[str, list[str]] = {
    "hidden_size": ["hidden_size", "d_model", "n_embd", "model_dim"],
    "num_hidden_layers": ["num_hidden_layers", "n_layer", "n_layers", "num_layers"],
    "num_attention_heads": ["num_attention_heads", "n_head", "n_heads", "num_heads"],
    "num_key_value_heads": ["num_key_value_heads", "num_kv_heads", "n_kv_heads"],
    "intermediate_size": ["intermediate_size", "ffn_dim", "inner_dim"],
    "max_position_embeddings": ["max_position_embeddings", "n_positions", "max_seq_len",
                                "max_sequence_length", "n_ctx"],
    "vocab_size": ["vocab_size"],
    "head_dim": ["head_dim"],
    "hidden_act": ["hidden_act", "activation_function", "hidden_activation"],
    "rms_norm_eps": ["rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon", "norm_eps"],
    "rope_theta": ["rope_theta", "rotary_emb_base", "rope_emb_base"],
    "torch_dtype": ["torch_dtype"],
    "attention_bias": ["attention_bias"],
    "tie_word_embeddings": ["tie_word_embeddings"],
    "qk_norm": ["qk_norm"],
    "sliding_window": ["sliding_window"],
    "num_experts": ["num_experts", "num_local_experts"],
    "num_experts_per_tok": ["num_experts_per_tok", "num_experts_per_token",
                            "experts_per_token", "top_k"],
    "moe_intermediate_size": ["moe_intermediate_size", "expert_intermediate_size", "ffn_dim_exps"],
    "shared_expert_intermediate_size": ["shared_expert_intermediate_size"],
    "num_shared_experts": ["num_shared_experts"],
}

# Bytes per element for common dtypes
DTYPE_BYTES: dict[str, float] = {
    "float32": 4, "float": 4,
    "float16": 2, "half": 2,
    "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
    "int8": 1,
    "int4": 0.5,
}

# Common GPU VRAM tiers in GB — used to recommend the smallest GPU that fits a model
GPU_TIERS = [4, 8, 12, 16, 24, 40, 48, 80, 141]
GPU_TIER_NAMES = {
    4: "entry consumer (4GB)",
    8: "RTX 3070/4060 (8GB)",
    12: "RTX 3080/4070 (12GB)",
    16: "RTX 3080 Ti/4080 (16GB)",
    24: "RTX 3090/4090 / A10 (24GB)",
    40: "A100 40GB",
    48: "A6000 / L40S (48GB)",
    80: "A100 80GB / H100 (80GB)",
    141: "H200 (141GB)",
}

TOKENS_PER_PAGE = 500   # rough estimate: 1 page ≈ 500 tokens
TOKENS_PER_WORD = 1.3   # rough estimate: 1 word ≈ 1.3 tokens

OVERHEAD_BYTES = 1 * 1024 ** 3  # 1 GB framework/activation overhead added to VRAM estimates

# Canonical fields that are important enough to flag when missing
_IMPORTANT_FIELDS = {
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "intermediate_size", "max_position_embeddings", "vocab_size",
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class VramEstimate:
    """VRAM requirements in bytes at different quantization levels."""
    weights_bf16: Optional[int] = None
    weights_int8: Optional[int] = None
    weights_int4: Optional[int] = None
    kv_max_ctx: Optional[int] = None    # KV cache at max_position_embeddings
    kv_seq_len: Optional[int] = None    # KV cache at user-specified seq_len
    total_bf16: Optional[int] = None    # weights_bf16 + kv_seq_len + overhead
    total_int8: Optional[int] = None
    total_int4: Optional[int] = None
    min_gpu_gb: Optional[int] = None    # Smallest GPU tier fitting int4 total


@dataclass
class ModelStats:
    """All architectural statistics and derived metrics for one model."""
    # Identity
    model_id: str
    model_type: str = "unknown"
    tokenizer_class: Optional[str] = None
    pos_enc: str = "?"                  # "RoPE", "ALiBi", "APE", "?"

    # Core dimensions
    vocab_size: Optional[int] = None
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None

    # Attention details
    attention_type: Optional[str] = None    # "MHA" / "MQA" / "GQA-Nx"
    attention_bias: Optional[bool] = None
    qk_norm: Optional[bool] = None
    sliding_window: Optional[int] = None

    # FFN / activation
    hidden_act: Optional[str] = None
    ffn_expansion_ratio: Optional[float] = None

    # Positional encoding
    rope_theta: Optional[float] = None

    # Misc
    torch_dtype: Optional[str] = None
    tie_word_embeddings: Optional[bool] = None
    rms_norm_eps: Optional[float] = None

    # MoE fields
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    shared_expert_intermediate_size: Optional[int] = None
    num_shared_experts: Optional[int] = None

    # Calculated parameter counts
    total_params: Optional[int] = None
    active_params: Optional[int] = None
    embed_params: Optional[int] = None
    attn_params: Optional[int] = None       # all attention layers combined
    ffn_params: Optional[int] = None        # all FFN layers combined
    lm_head_params: Optional[int] = None
    kv_cache_bytes_per_token: Optional[int] = None

    # VRAM estimates
    vram: Optional[VramEstimate] = None

    # Diagnostics
    fetch_error: Optional[str] = None
    missing_fields: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fetch Layer
# ---------------------------------------------------------------------------

def _hf_headers(token: Optional[str]) -> dict:
    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def fetch_hf_config(model_id: str, token: Optional[str] = None, revision: str = "main", timeout: int = 30) -> dict:
    """Download config.json from HuggingFace without downloading model weights."""
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/config.json"
    resp = requests.get(url, headers=_hf_headers(token), timeout=timeout)
    if resp.status_code == 401:
        raise requests.HTTPError("401 Unauthorized — model requires authentication. Pass --token or set HF_TOKEN.")
    if resp.status_code == 404:
        raise requests.HTTPError(f"404 Not Found — '{model_id}' not found on HuggingFace.")
    resp.raise_for_status()
    return resp.json()


def fetch_hf_tokenizer_config(model_id: str, token: Optional[str] = None, revision: str = "main", timeout: int = 30) -> dict:
    """Download tokenizer_config.json. Returns {} on any error (tokenizer info is optional)."""
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/tokenizer_config.json"
    try:
        resp = requests.get(url, headers=_hf_headers(token), timeout=timeout)
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return {}


def get_field(config: dict, canonical: str, default=None) -> tuple:
    """Look up a canonical field using FIELD_ALIASES. Returns (value, found_bool)."""
    for alias in FIELD_ALIASES.get(canonical, [canonical]):
        if alias in config:
            return config[alias], True
    return default, False


# ---------------------------------------------------------------------------
# Model-Type Quirk Overrides
# ---------------------------------------------------------------------------
# Called before field extraction. Return a dict of canonical-field overrides.
# Add a quirk only if important fields are absent or wrong (systematic structural deviations)

def _gpt2_overrides(config: dict) -> dict:
    """GPT-2/GPT-Neo: intermediate_size absent (=4×hidden), absolute positional embed."""
    h = config.get("n_embd", config.get("hidden_size", 0))
    overrides: dict = {"pos_enc": "APE", "rope_theta": None, "ffn_matrices": 2}
    if "intermediate_size" not in config and h:
        overrides["intermediate_size"] = 4 * h
    return overrides


def _bloom_overrides(config: dict) -> dict:
    """BLOOM uses ALiBi positional bias, no RoPE."""
    return {"pos_enc": "ALiBi", "rope_theta": None, "ffn_matrices": 2}


def _opt_overrides(config: dict) -> dict:
    """OPT uses learned absolute positional embeddings, no RoPE."""
    h = config.get("hidden_size", config.get("d_model", 0))
    overrides: dict = {"pos_enc": "APE", "rope_theta": None, "ffn_matrices": 2}
    if "intermediate_size" not in config and "ffn_dim" not in config and h:
        overrides["intermediate_size"] = 4 * h
    return overrides


def _phi_overrides(config: dict) -> dict:
    """Phi-2/Phi-3: partial_rotary_factor means only a fraction of head_dim gets RoPE.
    rope_theta is valid but the effective rotary dim = partial_rotary_factor × head_dim.
    This doesn't affect parameter counts, so no override needed for params.
    """
    return {}


def _gemma_overrides(config: dict) -> dict:
    """Gemma/Gemma2/Gemma3: 4 RMSNorm layers per block instead of the standard 2
    (pre-attn, post-attn, pre-FFN, post-FFN). Adds 2×hidden extra norms per layer.
    """
    return {"extra_norms_per_layer": 2}


QUIRK_OVERRIDES: dict = {
    "gpt2":    _gpt2_overrides,
    "gpt_neo": _gpt2_overrides,
    "bloom":   _bloom_overrides,
    "opt":     _opt_overrides,
    "phi":     _phi_overrides,
    "phi3":    _phi_overrides,
    "gemma":   _gemma_overrides,
    "gemma2":  _gemma_overrides,
    "gemma3":  _gemma_overrides,
}


_ALL_EXTRACTED_FIELDS = [
    "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "intermediate_size", "max_position_embeddings",
    "hidden_act", "rms_norm_eps", "rope_theta", "torch_dtype",
    "attention_bias", "tie_word_embeddings", "qk_norm", "sliding_window",
    "num_experts", "num_experts_per_tok", "moe_intermediate_size",
    "shared_expert_intermediate_size", "num_shared_experts",
]


def parse_config(model_id: str, config: dict, tokenizer_cfg: Optional[dict] = None) -> ModelStats:
    """Parse a raw HuggingFace config.json dict into a ModelStats object."""
    stats = ModelStats(model_id=model_id)
    stats.model_type = config.get("model_type", "unknown")

    if tokenizer_cfg:
        stats.tokenizer_class = tokenizer_cfg.get("tokenizer_class")

    # Apply quirk overrides for this model family
    quirk_fn = QUIRK_OVERRIDES.get(stats.model_type)
    quirks: dict = quirk_fn(config) if quirk_fn else {}

    def _get(canonical: str):
        if canonical in quirks:
            return quirks[canonical], True
        return get_field(config, canonical)

    # Extract all standard fields; track which important ones are missing
    missing = []
    for attr in _ALL_EXTRACTED_FIELDS:
        val, found = _get(attr)
        setattr(stats, attr, val)
        if not found and attr in _IMPORTANT_FIELDS:
            missing.append(attr)
    stats.missing_fields = missing

    # Positional encoding type
    stats.pos_enc = quirks.get("pos_enc", "RoPE" if stats.rope_theta else "?")

    # head_dim: use explicit value if present, else compute from hidden_size / num_heads
    explicit_hd, found = get_field(config, "head_dim")
    if found:
        stats.head_dim = explicit_hd
    elif stats.hidden_size and stats.num_attention_heads:
        stats.head_dim = stats.hidden_size // stats.num_attention_heads

    # Default KV heads = Q heads (standard MHA); overridden by GQA/MQA configs
    if stats.num_key_value_heads is None:
        stats.num_key_value_heads = stats.num_attention_heads

    # Attention type classification
    nh, nkv = stats.num_attention_heads, stats.num_key_value_heads
    if nh and nkv:
        if nkv == nh:
            stats.attention_type = "MHA"
        elif nkv == 1:
            stats.attention_type = "MQA"
        else:
            stats.attention_type = f"GQA-{nh // nkv}x"

    # FFN expansion ratio
    if stats.intermediate_size and stats.hidden_size:
        stats.ffn_expansion_ratio = stats.intermediate_size / stats.hidden_size

    # Parameter counts (analytical, no weights needed)
    result = _calculate_params(stats, quirks)
    (stats.total_params, stats.active_params, stats.embed_params,
     stats.attn_params, stats.ffn_params, stats.lm_head_params) = result

    # KV cache bytes per token
    stats.kv_cache_bytes_per_token = _calculate_kv_cache(stats)

    return stats


# ---------------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------------

def _calculate_params(stats: ModelStats, quirks: dict) -> tuple:
    """
    Analytically compute parameter counts from config fields. No weights downloaded.

    Returns (total, active, embed, attn_all_layers, ffn_all_layers, lm_head).
    Any element can be None if required fields are absent.

    Formula summary
    ---------------
    Embedding table:  vocab × hidden
    LM head:          vocab × hidden  (0 if tie_word_embeddings=True)

    Per attention layer:
        Q: hidden × (n_heads × head_dim)      + bias if attention_bias
        K: hidden × (n_kv_heads × head_dim)   + bias
        V: hidden × (n_kv_heads × head_dim)   + bias
        O: (n_heads × head_dim) × hidden      + bias
        Pre-attn RMSNorm: hidden
        QK-Norm (if qk_norm=True): (n_heads + n_kv_heads) × head_dim

    Per FFN layer (SwiGLU / 3-matrix, used by Llama/Mistral/Qwen/most modern models):
        gate + up + down: 2×hidden×ffn_dim + ffn_dim×hidden = 3×hidden×ffn_dim
        Pre-FFN RMSNorm: hidden

    Per FFN layer (GPT-2 / 2-matrix, for model_type in gpt2/gpt_neo/bloom/opt):
        up + down: 2×hidden×ffn_dim

    Per MoE layer:
        Router: hidden × num_experts
        Each expert (gate+up+down): 3 × hidden × moe_intermediate_size
        Shared expert (if present): n_shared × 3 × hidden × shared_intermediate_size

    Final RMSNorm: hidden
    """
    h = stats.hidden_size
    L = stats.num_hidden_layers
    nh = stats.num_attention_heads
    nkv = stats.num_key_value_heads
    dh = stats.head_dim
    vocab = stats.vocab_size
    ffn_dim = stats.intermediate_size

    if not all([h, L, nh, nkv, dh, vocab]):
        return (None,) * 6

    has_bias = stats.attention_bias or False
    has_qk_norm = stats.qk_norm or False
    tie_emb = stats.tie_word_embeddings if stats.tie_word_embeddings is not None else False
    extra_norms = quirks.get("extra_norms_per_layer", 0)
    ffn_matrices = quirks.get("ffn_matrices", 3)  # 3=SwiGLU, 2=standard

    # Embedding table
    embed_p = vocab * h

    # LM head (separate weights unless tied)
    lm_head_p = 0 if tie_emb else vocab * h

    # Per-layer attention
    q_p = h * (nh * dh)  + (nh * dh  if has_bias else 0)
    k_p = h * (nkv * dh) + (nkv * dh if has_bias else 0)
    v_p = h * (nkv * dh) + (nkv * dh if has_bias else 0)
    o_p = (nh * dh) * h  + (h if has_bias else 0)
    attn_norm_p = h
    qk_norm_p = (nh + nkv) * dh if has_qk_norm else 0
    per_attn = q_p + k_p + v_p + o_p + attn_norm_p + qk_norm_p

    # Per-layer FFN
    is_moe = bool(stats.num_experts and stats.num_experts > 1)
    ffn_norm_p = h
    extra_norm_p = extra_norms * h

    if is_moe:
        E = stats.num_experts
        moe_ffn = stats.moe_intermediate_size
        if not moe_ffn:
            return (None,) * 6
        router_p = h * E
        all_expert_p = E * (3 * h * moe_ffn)     # gate + up + down per expert
        shared_p = 0
        if stats.shared_expert_intermediate_size:
            ns = stats.num_shared_experts or 1
            shared_p = ns * (3 * h * stats.shared_expert_intermediate_size)
        per_ffn = router_p + all_expert_p + shared_p + ffn_norm_p + extra_norm_p
    else:
        if not ffn_dim:
            return (None,) * 6
        if ffn_matrices == 3:
            # SwiGLU/GeGLU: gate_proj + up_proj + down_proj
            per_ffn = (2 * h * ffn_dim) + (ffn_dim * h) + ffn_norm_p + extra_norm_p
        else:
            # Standard 2-matrix FFN (GPT-2, BLOOM, OPT)
            per_ffn = (h * ffn_dim) + (ffn_dim * h) + ffn_norm_p + extra_norm_p

    final_norm_p = h

    attn_total = L * per_attn
    ffn_total = L * per_ffn
    total = embed_p + lm_head_p + attn_total + ffn_total + final_norm_p

    # For MoE, active params substitute the active expert count for all experts
    if is_moe and stats.num_experts_per_tok:
        k = stats.num_experts_per_tok
        moe_ffn = stats.moe_intermediate_size
        active_expert_p = k * (3 * h * moe_ffn)
        shared_p2 = 0
        if stats.shared_expert_intermediate_size:
            ns = stats.num_shared_experts or 1
            shared_p2 = ns * (3 * h * stats.shared_expert_intermediate_size)
        active_per_ffn = router_p + active_expert_p + shared_p2 + ffn_norm_p + extra_norm_p
        active_total = embed_p + lm_head_p + (L * per_attn) + (L * active_per_ffn) + final_norm_p
    else:
        active_total = total

    return total, active_total, embed_p, attn_total, ffn_total, lm_head_p


def _calculate_kv_cache(stats: ModelStats) -> Optional[int]:
    """KV cache size in bytes per token (K + V across all layers)."""
    if not all([stats.num_key_value_heads, stats.head_dim, stats.num_hidden_layers]):
        return None
    dtype_b = DTYPE_BYTES.get(stats.torch_dtype or "bfloat16", 2)
    return int(2 * stats.num_key_value_heads * stats.head_dim * stats.num_hidden_layers * dtype_b)


def calculate_vram(stats: ModelStats, seq_len: Optional[int] = None) -> VramEstimate:
    """
    Estimate VRAM at three quantization levels.

    Formula (per level):
        Weights  = total_params × bytes_per_param  (2 / 1 / 0.5 for bf16 / int8 / int4)
        KV cache = kv_cache_per_token × seq_len    (kept at bf16 even when weights quantized)
        Total    = Weights + KV cache + 1 GB overhead

    Reference: modal.com/blog/how-much-vram-need-inference
    """
    v = VramEstimate()
    if stats.total_params is None:
        return v

    p = stats.total_params
    v.weights_bf16 = p * 2
    v.weights_int8 = p * 1
    v.weights_int4 = p // 2    # ~0.5 bytes/param

    kv_per_tok = stats.kv_cache_bytes_per_token
    if kv_per_tok and stats.max_position_embeddings:
        v.kv_max_ctx = kv_per_tok * stats.max_position_embeddings
        effective_seq = seq_len if seq_len else stats.max_position_embeddings
        v.kv_seq_len = kv_per_tok * effective_seq
    else:
        v.kv_seq_len = 0

    kv = v.kv_seq_len or 0
    v.total_bf16 = v.weights_bf16 + kv + OVERHEAD_BYTES
    v.total_int8 = v.weights_int8 + kv + OVERHEAD_BYTES
    v.total_int4 = v.weights_int4 + kv + OVERHEAD_BYTES

    int4_gb = v.total_int4 / 1024 ** 3
    for tier in GPU_TIERS:
        if tier >= int4_gb:
            v.min_gpu_gb = tier
            break

    return v


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def fmt_number(n, decimals: int = 1) -> str:
    """Format a large integer with B/M/K suffix. Returns '?' for None."""
    if n is None:
        return "?"
    n = int(n)
    if n >= 1_000_000_000_000:
        return f"{n / 1e12:.{decimals}f}T"
    if n >= 1_000_000_000:
        return f"{n / 1e9:.{decimals}f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.{decimals}f}M"
    if n >= 1_000:
        return f"{n / 1e3:.{decimals}f}K"
    return str(n)


def fmt_context(n) -> str:
    """Format context length as e.g. '128K'. Returns '?' for None."""
    if n is None:
        return "?"
    n = int(n)
    k = n / 1024
    if k == int(k):
        return f"{int(k)}K"
    return f"{k:.0f}K"


def fmt_gb(n) -> str:
    """Format bytes as GB string."""
    if n is None:
        return "?"
    return f"{n / 1024 ** 3:.1f}GB"


def fmt_kb(n) -> str:
    """Format bytes as KB or MB string."""
    if n is None:
        return "?"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.0f}MB"
    return f"{n / 1024:.0f}KB"


def fmt_pct(num, denom) -> str:
    """Format num/denom as a percentage string."""
    if num is None or denom is None or denom == 0:
        return "?"
    return f"{100 * num / denom:.0f}%"


def _short_id(model_id: str) -> str:
    """Return last 1-2 path components of a HuggingFace model ID."""
    parts = model_id.strip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else model_id


def _make_table(headers: list[str], rows: list[list]) -> str:
    """Build a plain-text auto-sized comparison table."""
    n_cols = len(headers)
    padded = [list(row) + [""] * (n_cols - len(row)) for row in rows]
    all_rows = [[str(h) for h in headers]] + [[str(c) for c in r] for r in padded]
    widths = [max(len(cell) for cell in col) for col in zip(*all_rows)]
    sep = "  ".join("-" * w for w in widths)

    def fmt_row(row):
        return "  ".join(str(c).ljust(w) for c, w in zip(row, widths))

    return "\n".join([fmt_row(headers), sep] + [fmt_row(r) for r in padded])


# ---------------------------------------------------------------------------
# Table Formatters
# ---------------------------------------------------------------------------

def format_arch_table(stats_list: list[ModelStats]) -> str:
    headers = [
        "Model", "Type", "Tokenizer", "Vocab", "Context", "Layers",
        "Heads", "KV-Heads", "HeadDim", "FFN-Size", "FFN-Ratio",
        "Attn-Type", "Activation", "QK-Norm", "Pos-Enc", "RoPE-Base",
        "Dtype", "KV-Cache/tok",
    ]
    rows = []
    for s in stats_list:
        if s.fetch_error:
            rows.append([_short_id(s.model_id), f"ERROR: {s.fetch_error}"])
            continue
        rope_str = (
            f"{s.rope_theta:,.0f}" if s.rope_theta
            else ("N/A" if s.pos_enc in ("ALiBi", "APE") else "?")
        )
        qk_str = ("Yes" if s.qk_norm else "No") if s.qk_norm is not None else "?"
        rows.append([
            _short_id(s.model_id),
            s.model_type or "?",
            s.tokenizer_class or "?",
            fmt_number(s.vocab_size),
            fmt_context(s.max_position_embeddings),
            str(s.num_hidden_layers or "?"),
            str(s.num_attention_heads or "?"),
            str(s.num_key_value_heads or "?"),
            str(s.head_dim or "?"),
            fmt_number(s.intermediate_size),
            f"{s.ffn_expansion_ratio:.1f}x" if s.ffn_expansion_ratio else "?",
            s.attention_type or "?",
            s.hidden_act or "?",
            qk_str,
            s.pos_enc,
            rope_str,
            s.torch_dtype or "?",
            fmt_kb(s.kv_cache_bytes_per_token),
        ])
    return "=== Architecture ===\n" + _make_table(headers, rows)


def format_param_table(stats_list: list[ModelStats]) -> str:
    headers = [
        "Model", "Total", "Active",
        "Embed", "Embed%", "Attn (all layers)", "Attn%",
        "FFN (all layers)", "FFN%", "LM-Head", "LM-Head%",
    ]
    rows = []
    for s in stats_list:
        if s.fetch_error:
            rows.append([_short_id(s.model_id), "ERROR"])
            continue
        active_str = "—" if s.active_params == s.total_params else fmt_number(s.active_params)
        rows.append([
            _short_id(s.model_id),
            fmt_number(s.total_params),
            active_str,
            fmt_number(s.embed_params),
            fmt_pct(s.embed_params, s.total_params),
            fmt_number(s.attn_params),
            fmt_pct(s.attn_params, s.total_params),
            fmt_number(s.ffn_params),
            fmt_pct(s.ffn_params, s.total_params),
            fmt_number(s.lm_head_params),
            fmt_pct(s.lm_head_params, s.total_params),
        ])
    return "=== Parameter Breakdown ===\n" + _make_table(headers, rows)


def format_vram_table(stats_list: list[ModelStats], seq_len: Optional[int] = None) -> str:
    has_custom = seq_len is not None
    if has_custom:
        kv_col = f"KV@{seq_len // 1000}K-tok" if seq_len >= 1000 else f"KV@{seq_len}-tok"
    else:
        kv_col = "KV@MaxCtx"

    base_headers = ["Model", "Weights-bf16", "Weights-int8", "Weights-int4", "KV@MaxCtx"]
    if has_custom:
        base_headers.append(kv_col)
    base_headers += ["Total-bf16", "Total-int8", "Total-int4", "Min-GPU (int4)"]

    rows = []
    for s in stats_list:
        if s.fetch_error:
            rows.append([_short_id(s.model_id), "ERROR"])
            continue
        v = s.vram
        if v is None:
            rows.append([_short_id(s.model_id)] + ["?"] * (len(base_headers) - 1))
            continue
        row = [
            _short_id(s.model_id),
            fmt_gb(v.weights_bf16),
            fmt_gb(v.weights_int8),
            fmt_gb(v.weights_int4),
            fmt_gb(v.kv_max_ctx),
        ]
        if has_custom:
            row.append(fmt_gb(v.kv_seq_len))
        row += [
            fmt_gb(v.total_bf16),
            fmt_gb(v.total_int8),
            fmt_gb(v.total_int4),
            (f"{v.min_gpu_gb}GB" if v.min_gpu_gb else ">141GB"),
        ]
        rows.append(row)

    seq_note = f" (KV cache at {seq_len:,} tokens via --seq-len)" if has_custom else " (KV cache at max context length)"
    footer = "\nNote: KV cache stays bf16 even when weights are quantized. +1GB overhead included in totals."
    return f"=== VRAM Estimates{seq_note} ===\n" + _make_table(base_headers, rows) + footer


def generate_deductions(stats: ModelStats, seq_len: Optional[int] = None) -> str:
    """Generate a plain-English explanation of one model's architectural choices."""
    if stats.fetch_error:
        return f"── {stats.model_id} ──\n  ERROR: {stats.fetch_error}"

    lines = []
    sep = "─" * max(2, 70 - len(stats.model_id) - 4)
    lines.append(f"── {stats.model_id} {sep}")

    def row(label: str, value: str):
        lines.append(f"  {label:<15}{value}")

    def note(value: str):
        lines.append(f"  {'':15}{value}")

    # Layer structure
    if stats.num_hidden_layers and stats.attn_params is not None and stats.ffn_params is not None:
        layer_p = (stats.attn_params + stats.ffn_params) // stats.num_hidden_layers
        total_layer_p = stats.attn_params + stats.ffn_params
        row("Structure:", f"{stats.num_hidden_layers} layers × {fmt_number(layer_p)} params/layer = {fmt_number(total_layer_p)} total layer params")

    # Parameter budget
    if stats.total_params and all(x is not None for x in [stats.embed_params, stats.attn_params, stats.ffn_params, stats.lm_head_params]):
        t = stats.total_params
        tied = "(shared weights, no extra params)" if stats.tie_word_embeddings else ""
        row("Budget:", (
            f"Embed {fmt_pct(stats.embed_params, t)} | "
            f"Attn {fmt_pct(stats.attn_params, t)} | "
            f"FFN {fmt_pct(stats.ffn_params, t)} | "
            f"LM-head {fmt_pct(stats.lm_head_params, t)} {tied}"
        ))

    # Attention mechanism
    if stats.attention_type and stats.num_attention_heads and stats.num_key_value_heads:
        nh, nkv, dh = stats.num_attention_heads, stats.num_key_value_heads, stats.head_dim
        if stats.attention_type == "MHA":
            attn_desc = f"MHA — {nh} Q-heads = {nkv} KV-heads (full KV cache, one entry per head)"
        elif stats.attention_type == "MQA":
            attn_desc = f"MQA — {nh} Q-heads share 1 KV-head; KV cache is {nh}× smaller than MHA"
        else:
            ratio = nh // nkv
            attn_desc = f"{stats.attention_type} — {nh} Q-heads share {nkv} KV-heads; KV cache is {ratio}× smaller than MHA"
        row("Attention:", attn_desc)
        if dh:
            kv_vals = 2 * nkv * dh
            note(f"head_dim={dh}; each layer caches {kv_vals:,} values/token (K + V combined)")

    # QK-Norm
    if stats.qk_norm:
        row("QK-Norm:", "Yes — Q and K vectors are normalized per-head before dot-product (stabilizes training at scale)")

    # Sliding window attention
    if stats.sliding_window:
        row("Sliding win:", f"{stats.sliding_window:,} tokens — each token attends only to a local window (O(n) memory vs O(n²))")

    # FFN
    if stats.hidden_act and stats.intermediate_size and stats.hidden_size:
        act = stats.hidden_act.lower()
        ratio = stats.ffn_expansion_ratio or 0
        ffn_s = stats.intermediate_size
        h = stats.hidden_size
        if act in ("silu", "swish"):
            ffn_desc = f"SwiGLU — 3 matrices (gate+up+down), {ratio:.1f}× expansion ({ffn_s:,} / {h:,})"
            ffn_note = "gate projected through SiLU acts as a gating signal on the up-projection before the down-projection"
        elif act in ("gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh"):
            ffn_desc = f"GeGLU/GELU — 3 matrices (gate+up+down), {ratio:.1f}× expansion ({ffn_s:,} / {h:,})"
            ffn_note = ""
        else:
            ffn_desc = f"{act.upper()} — 2 matrices (up+down), {ratio:.1f}× expansion ({ffn_s:,} / {h:,})"
            ffn_note = ""
        row("FFN:", ffn_desc)
        if ffn_note:
            note(ffn_note)

    # MoE
    if stats.num_experts and stats.num_experts > 1:
        k = stats.num_experts_per_tok
        pct = f"{100 * k / stats.num_experts:.1f}%" if k else "?"
        row("MoE:", f"{stats.num_experts} total experts, {k or '?'} active per token ({pct} compute utilization per layer)")
        if stats.active_params and stats.total_params and stats.active_params != stats.total_params:
            note(f"Active = {fmt_number(stats.active_params)} / Total = {fmt_number(stats.total_params)} — {fmt_pct(stats.active_params, stats.total_params)} of weights engaged per forward pass")
        if stats.moe_intermediate_size and stats.hidden_size:
            moe_ratio = stats.moe_intermediate_size / stats.hidden_size
            note(f"Per-expert FFN: {stats.moe_intermediate_size:,} intermediate dim ({moe_ratio:.2f}× expansion, smaller than dense FFN by design)")
    else:
        if stats.num_hidden_layers:
            row("MoE:", f"Dense — all {stats.num_hidden_layers} FFN layers are always active (100% compute utilization)")

    # Context window
    if stats.max_position_embeddings:
        ctx = stats.max_position_embeddings
        words = ctx / TOKENS_PER_WORD
        pages = ctx / TOKENS_PER_PAGE
        row("Context:", f"{ctx:,} tokens ≈ {words:,.0f} words ≈ {pages:,.0f} pages (at ~500 tok/page)")
        if stats.rope_theta:
            if stats.rope_theta >= 500_000:
                rope_note = "very high RoPE base → explicitly designed for long-context extrapolation"
            elif stats.rope_theta >= 100_000:
                rope_note = "high RoPE base → extended context training beyond default 8K"
            elif stats.rope_theta >= 10_000:
                rope_note = "standard-to-moderate RoPE base"
            else:
                rope_note = "low RoPE base — may not generalize well beyond training context length"
            note(f"RoPE base {stats.rope_theta:,.0f} — {rope_note}")
        elif stats.pos_enc in ("ALiBi", "APE"):
            note(f"Positional encoding: {stats.pos_enc} (not RoPE; long-context behavior differs from RoPE-based models)")

    # KV cache growth rate
    if stats.kv_cache_bytes_per_token and stats.max_position_embeddings:
        kb_per_tok = stats.kv_cache_bytes_per_token / 1024
        full_gb = stats.kv_cache_bytes_per_token * stats.max_position_embeddings / 1024 ** 3
        dtype = stats.torch_dtype or "bf16"
        row("KV cache:", f"{kb_per_tok:.0f} KB/token → {full_gb:.1f} GB at full {fmt_context(stats.max_position_embeddings)} context ({dtype})")
        if seq_len and seq_len != stats.max_position_embeddings:
            seq_gb = stats.kv_cache_bytes_per_token * seq_len / 1024 ** 3
            note(f"{seq_gb:.1f} GB at {seq_len:,} tokens (--seq-len)")

    # VRAM quick summary
    if stats.vram and stats.vram.total_bf16:
        v = stats.vram
        kv_ref = v.kv_seq_len if (seq_len and v.kv_seq_len) else v.kv_max_ctx
        row("VRAM (bf16):", f"~{fmt_gb(v.total_bf16)} total = {fmt_gb(v.weights_bf16)} weights + {fmt_gb(kv_ref)} KV + 1GB overhead")
        if v.min_gpu_gb:
            gpu_name = GPU_TIER_NAMES.get(v.min_gpu_gb, f"{v.min_gpu_gb}GB GPU")
            note(f"Minimum GPU to run at int4: {v.min_gpu_gb}GB — {gpu_name}")
            note(f"int4 total = {fmt_gb(v.total_int4)} ({fmt_gb(v.weights_int4)} weights + {fmt_gb(kv_ref)} KV + 1GB overhead)")

    # Missing fields notice
    if stats.missing_fields:
        row("⚠ Missing:", ", ".join(stats.missing_fields) + " — shown as '?' in tables above")

    return "\n".join(lines)


def format_deductions(stats_list: list[ModelStats], seq_len: Optional[int] = None) -> str:
    blocks = [generate_deductions(s, seq_len) for s in stats_list]
    return "=== Model Insights ===\n" + "\n\n".join(blocks)


def format_json(stats_list: list[ModelStats]) -> str:
    return json.dumps([asdict(s) for s in stats_list], indent=2, default=str)


def format_csv(stats_list: list[ModelStats]) -> str:
    cols = [
        "model_id", "model_type", "tokenizer_class", "pos_enc",
        "vocab_size", "max_position_embeddings", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "hidden_size", "intermediate_size", "ffn_expansion_ratio", "hidden_act",
        "attention_type", "qk_norm", "attention_bias", "rope_theta", "torch_dtype",
        "sliding_window", "num_experts", "num_experts_per_tok", "moe_intermediate_size",
        "total_params", "active_params", "embed_params", "attn_params", "ffn_params", "lm_head_params",
        "kv_cache_bytes_per_token",
        "vram_weights_bf16", "vram_weights_int8", "vram_weights_int4",
        "vram_kv_max_ctx", "vram_kv_seq_len",
        "vram_total_bf16", "vram_total_int8", "vram_total_int4", "vram_min_gpu_gb",
        "fetch_error", "missing_fields",
    ]

    def val(s: ModelStats, col: str) -> str:
        if col.startswith("vram_"):
            v = s.vram
            if v is None:
                return ""
            return str(getattr(v, col[5:], "") or "")
        if col == "missing_fields":
            return ";".join(s.missing_fields)
        v = getattr(s, col, "")
        return "" if v is None else str(v)

    lines = [",".join(cols)]
    for s in stats_list:
        lines.append(",".join(f'"{val(s, c)}"' for c in cols))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch and compare LLM architecture stats from HuggingFace (no weight download).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llm_stats.py Qwen/Qwen3-1.7B meta-llama/Llama-3.2-1B mistralai/Mistral-7B-v0.3
  python llm_stats.py --seq-len 4096 mistralai/Mistral-7B-v0.3
  python llm_stats.py --format json Qwen/Qwen3-1.7B > stats.json
  python llm_stats.py --format csv model1 model2 > comparison.csv
  python llm_stats.py --dump-raw new-org/new-model
  python llm_stats.py --token $HF_TOKEN meta-llama/Llama-3.1-8B

Adding support for a new model type:
  1. Run with --dump-raw to inspect the raw config.json
  2. If a field uses a non-standard name, add it to FIELD_ALIASES in this file
  3. If the model has a structural difference (e.g. absent intermediate_size,
     non-RoPE positional encoding), add a small _xxx_overrides() function
     and register it in QUIRK_OVERRIDES
""",
    )
    parser.add_argument("models", nargs="+", help="HuggingFace model IDs (e.g. Qwen/Qwen3-1.7B)")
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table", help="Output format (default: table)")
    parser.add_argument("--token", help="HuggingFace access token for gated models (or set HF_TOKEN env var)")
    parser.add_argument("--revision", default="main", help="Git revision: branch, tag, or commit SHA (default: main)")
    parser.add_argument("--seq-len", type=int, default=None, metavar="N", help="Sequence length for VRAM estimate (default: model's max context)")
    parser.add_argument("--dump-raw", action="store_true", help="Print raw config.json for each model and exit")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP request timeout in seconds (default: 30)")
    args = parser.parse_args()

    all_stats: list[ModelStats] = []

    for model_id in args.models:
        print(f"Fetching {model_id} ...", end=" ", flush=True, file=sys.stderr)
        try:
            config = fetch_hf_config(model_id, token=args.token, revision=args.revision, timeout=args.timeout)

            if args.dump_raw:
                print("ok", file=sys.stderr)
                print(f"\n{'=' * 60}")
                print(f"  {model_id}  —  config.json")
                print(f"{'=' * 60}")
                print(json.dumps(config, indent=2))
                continue

            tok_cfg = fetch_hf_tokenizer_config(model_id, token=args.token, revision=args.revision, timeout=args.timeout)
            stats = parse_config(model_id, config, tok_cfg)
            stats.vram = calculate_vram(stats, seq_len=args.seq_len)
            print("ok", file=sys.stderr)

        except requests.HTTPError as e:
            print("FAILED", file=sys.stderr)
            stats = ModelStats(model_id=model_id, fetch_error=str(e))
        except Exception as e:
            print("FAILED", file=sys.stderr)
            stats = ModelStats(model_id=model_id, fetch_error=f"{type(e).__name__}: {e}")

        all_stats.append(stats)

    if args.dump_raw:
        return

    if not all_stats:
        sys.exit(0)

    if args.format == "json":
        print(format_json(all_stats))
    elif args.format == "csv":
        print(format_csv(all_stats))
    else:
        print()
        print(format_arch_table(all_stats))
        print()
        print(format_param_table(all_stats))
        print()
        print(format_vram_table(all_stats, seq_len=args.seq_len))
        print()
        print(format_deductions(all_stats, seq_len=args.seq_len))

        # Footer: list missing fields across all models
        all_missing: set[str] = set()
        for s in all_stats:
            all_missing.update(s.missing_fields)
        if all_missing:
            print(f"\n⚠  Fields not found in config.json: {', '.join(sorted(all_missing))}")
            print("   Run --dump-raw to inspect the raw config and add aliases if needed.")

    if all(s.fetch_error for s in all_stats):
        sys.exit(1)


if __name__ == "__main__":
    main()
