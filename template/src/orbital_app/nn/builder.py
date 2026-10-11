"""
Neural Network Builder - Architecture Tree → PyTorch Module

Compiles architecture S-expressions to PyTorch modules.
Supports both nn/ prefixed names (legacy) and prefix-less names (new).
Supports positional args and dict-style params.

Layers: linear, conv1d/2d/3d, conv-transpose2d, lstm, gru,
        multi-head-attention, embedding, bilinear,
        activations (relu, gelu, silu, tanh, sigmoid, softmax, leaky-relu, elu, mish, swiglu),
        norms (batch-norm, layer-norm, group-norm, instance-norm, rms-norm),
        regularization (dropout, dropout2d, alpha-dropout),
        pooling (max-pool2d, avg-pool2d, adaptive-avg-pool, global-mean-pool, global-max-pool),
        shape (flatten, reshape, select-token, select-hidden)

Composition: sequential, parallel, residual, repeat

Organisms: transformer-encoder-block, transformer-decoder-block, mlp, unet
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any, Dict, Optional, Union


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_param(layer: Union[List[Any], Dict[str, Any]], key_or_index, default=None):
    """Extract param from positional args (list) or dict-style params."""
    if isinstance(layer, dict):
        return layer.get(key_or_index, default)
    if isinstance(layer, list):
        if isinstance(key_or_index, int):
            return layer[key_or_index] if len(layer) > key_or_index else default
        return default
    return default


def _normalize_op(op: str) -> str:
    """Strip nn/ prefix for backward compatibility."""
    if op.startswith("nn/"):
        return op[3:]
    return op


def _parse_layer_spec(spec: Any) -> tuple:
    """Parse a layer spec into (op_name, params).
    Supports:
      ["linear", 16, 64]              -> ("linear", [16, 64])
      ["conv2d", {"in-channels": 3}]  -> ("conv2d", {"in-channels": 3})
      ["relu"]                         -> ("relu", [])
    """
    if isinstance(spec, list) and len(spec) >= 1:
        op = _normalize_op(str(spec[0]))
        args = spec[1:]
        if len(args) == 1 and isinstance(args[0], dict):
            return op, args[0]
        return op, args
    raise ValueError(f"Invalid layer spec: {spec}")


def _p(params, key, index, default=None):
    """Get param by key (dict) or index (list)."""
    if isinstance(params, dict):
        return params.get(key, default)
    if isinstance(params, list):
        return params[index] if len(params) > index else default
    return default


# ---------------------------------------------------------------------------
# Custom Modules
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMS Normalization (used in LLaMA)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation (used in LLaMA FFN)."""
    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ff_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GlobalMeanPool(nn.Module):
    """Global mean pooling over spatial dimensions."""
    def forward(self, x):
        if x.dim() == 3:
            return x.mean(dim=1)
        return x.mean(dim=list(range(2, x.dim())))


class GlobalMaxPool(nn.Module):
    """Global max pooling over spatial dimensions."""
    def forward(self, x):
        if x.dim() == 3:
            return x.max(dim=1).values
        dims = list(range(2, x.dim()))
        for d in reversed(dims):
            x = x.max(dim=d).values
        return x


class Reshape(nn.Module):
    """Reshape tensor (batch dim preserved)."""
    def __init__(self, shape: List[int]):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class SelectToken(nn.Module):
    """Select a single token from sequence dimension."""
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[:, self.index]


class SelectHidden(nn.Module):
    """Select hidden state from recurrent output."""
    def __init__(self, method: str = "last"):
        super().__init__()
        self.method = method

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        if self.method == "last":
            return x[:, -1]
        elif self.method == "mean":
            return x.mean(dim=1)
        elif self.method == "max":
            return x.max(dim=1).values
        raise ValueError(f"Unknown select-hidden method: {self.method}")


class Pooler(nn.Module):
    """Pool sequence output to single vector."""
    def __init__(self, method: str, dim: int):
        super().__init__()
        self.method = method
        if method == "cls-token":
            self.dense = nn.Linear(dim, dim)
            self.activation = nn.Tanh()

    def forward(self, x):
        if self.method == "cls-token":
            return self.activation(self.dense(x[:, 0]))
        elif self.method == "mean":
            return x.mean(dim=1)
        elif self.method == "max":
            return x.max(dim=1).values
        raise ValueError(f"Unknown pooler method: {self.method}")


class PositionalEncoding(nn.Module):
    """Positional encoding (sinusoidal, learned, or RoPE placeholder)."""
    def __init__(self, dim: int, max_len: int = 5000, pe_type: str = "sinusoidal"):
        super().__init__()
        self.pe_type = pe_type
        if pe_type == "learned":
            self.pe = nn.Embedding(max_len, dim)
        elif pe_type == "sinusoidal":
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:dim // 2] if dim % 2 else div_term)
            self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        if self.pe_type == "learned":
            positions = torch.arange(seq_len, device=x.device)
            return x + self.pe(positions)
        return x + self.pe[:, :seq_len]


class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dim (ViT)."""
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (image_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PrependCLSToken(nn.Module):
    """Prepend a learnable CLS token to the sequence."""
    def __init__(self, dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        cls = self.cls_token.expand(x.size(0), -1, -1)
        return torch.cat([cls, x], dim=1)


class TokenTypeEmbedding(nn.Module):
    """Segment embedding for sentence pair tasks (BERT)."""
    def __init__(self, num_types: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_types, dim)

    def forward(self, x, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros(x.size(0), x.size(1), dtype=torch.long, device=x.device)
        return x + self.emb(token_type_ids)


# ---------------------------------------------------------------------------
# Composition Modules
# ---------------------------------------------------------------------------

class ParallelBranches(nn.Module):
    """Run multiple branches on same input, merge results."""
    def __init__(self, branches: List[nn.Module], merge: str = "concat", names: Optional[List[str]] = None):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.merge = merge
        self.names = names

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        if self.merge == "none":
            if self.names:
                return {name: out for name, out in zip(self.names, outputs)}
            return outputs
        if self.merge == "concat":
            return torch.cat(outputs, dim=-1)
        if self.merge == "add":
            return sum(outputs)
        if self.merge == "mul":
            result = outputs[0]
            for o in outputs[1:]:
                result = result * o
            return result
        if self.merge == "mean":
            return torch.stack(outputs).mean(dim=0)
        if self.merge == "max":
            return torch.stack(outputs).max(dim=0).values
        raise ValueError(f"Unknown merge mode: {self.merge}")


class ResidualBlock(nn.Module):
    """Skip connection: output = block(x) + shortcut(x)."""
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut or nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


# ---------------------------------------------------------------------------
# Transformer Blocks (Organisms)
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block: self-attention + FFN + residuals + norms."""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, activation: str = "relu",
                 norm: str = "post", norm_type: str = "layer-norm",
                 causal_mask: bool = False, **kwargs):
        super().__init__()
        self.pre_norm = norm == "pre"
        self.causal_mask = causal_mask

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        if activation == "swiglu":
            self.ffn = SwiGLU(embed_dim, ff_dim)
        else:
            act_fn = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}.get(activation, nn.ReLU())
            self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), act_fn, nn.Dropout(dropout), nn.Linear(ff_dim, embed_dim))

        norm_cls = RMSNorm if norm_type == "rms-norm" else nn.LayerNorm
        self.norm1 = norm_cls(embed_dim)
        self.norm2 = norm_cls(embed_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if self.causal_mask and mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)

        if self.pre_norm:
            x2 = self.norm1(x)
            x = x + self.drop1(self.attn(x2, x2, x2, attn_mask=mask)[0])
            x2 = self.norm2(x)
            x = x + self.drop2(self.ffn(x2))
        else:
            x = self.norm1(x + self.drop1(self.attn(x, x, x, attn_mask=mask)[0]))
            x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class TransformerDecoderBlock(nn.Module):
    """Single transformer decoder block: causal self-attn + optional cross-attn + FFN."""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, cross_attention: bool = False,
                 activation: str = "relu", norm: str = "post",
                 norm_type: str = "layer-norm", **kwargs):
        super().__init__()
        self.pre_norm = norm == "pre"
        self.cross_attention = cross_attention

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        if cross_attention:
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            norm_cls = RMSNorm if norm_type == "rms-norm" else nn.LayerNorm
            self.norm_cross = norm_cls(embed_dim)
            self.drop_cross = nn.Dropout(dropout)

        if activation == "swiglu":
            self.ffn = SwiGLU(embed_dim, ff_dim)
        else:
            act_fn = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}.get(activation, nn.ReLU())
            self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), act_fn, nn.Dropout(dropout), nn.Linear(ff_dim, embed_dim))

        norm_cls = RMSNorm if norm_type == "rms-norm" else nn.LayerNorm
        self.norm1 = norm_cls(embed_dim)
        self.norm2 = norm_cls(embed_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, memory=None, mask=None):
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)

        if self.pre_norm:
            x2 = self.norm1(x)
            x = x + self.drop1(self.self_attn(x2, x2, x2, attn_mask=causal_mask)[0])
            if self.cross_attention and memory is not None:
                x2 = self.norm_cross(x)
                x = x + self.drop_cross(self.cross_attn(x2, memory, memory)[0])
            x2 = self.norm2(x)
            x = x + self.drop2(self.ffn(x2))
        else:
            x = self.norm1(x + self.drop1(self.self_attn(x, x, x, attn_mask=causal_mask)[0]))
            if self.cross_attention and memory is not None:
                x = self.norm_cross(x + self.drop_cross(self.cross_attn(x, memory, memory)[0]))
            x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


# ---------------------------------------------------------------------------
# Main Build Functions
# ---------------------------------------------------------------------------

def build_network(architecture: Any) -> nn.Module:
    """
    Build PyTorch module from architecture S-expression tree.

    Supports:
      ["sequential", ...layers]
      ["parallel", { "branches": [...], "merge": "concat" }]
      ["residual", { "block": [...], "shortcut": "identity" }]
      ["repeat", N, { "block": [...] }]
      Legacy: ["nn/sequential", ...]

    Returns:
        PyTorch nn.Module
    """
    if not architecture:
        raise ValueError("Empty architecture")

    if isinstance(architecture, dict):
        # Dict-form for organisms like unet, conditional-unet
        if "encoder" in architecture and "decoder" in architecture:
            return _build_organism_from_dict(architecture)
        raise ValueError(f"Unknown architecture dict format: {list(architecture.keys())}")

    op, params = _parse_layer_spec(architecture)

    if op == "sequential":
        layers_specs = params if isinstance(params, list) else params.get("layers", [])
        layers = [build_layer_or_network(spec) for spec in layers_specs]
        return nn.Sequential(*layers)

    if op == "parallel":
        p = params if isinstance(params, dict) else (params[0] if params and isinstance(params[0], dict) else {})
        branches = [build_network(b) for b in p.get("branches", [])]
        return ParallelBranches(branches, merge=p.get("merge", "concat"), names=p.get("names"))

    if op == "residual":
        p = params if isinstance(params, dict) else (params[0] if params and isinstance(params[0], dict) else {})
        block = build_network(p["block"]) if "block" in p else build_network(params)
        shortcut = None
        if p.get("shortcut") == "projection":
            shortcut = None  # Auto-projection handled at higher level
        return ResidualBlock(block, shortcut)

    if op == "repeat":
        if isinstance(params, list):
            count = params[0]
            block_spec = params[1] if len(params) > 1 else {}
        else:
            count = params.get("count", 1)
            block_spec = params
        block_def = block_spec.get("block", block_spec) if isinstance(block_spec, dict) else block_spec
        share = block_spec.get("share-weights", False) if isinstance(block_spec, dict) else False

        if share:
            block = build_network(block_def) if isinstance(block_def, list) else build_layer_or_network(block_def)
            layers = [block] * count
        else:
            layers = [build_network(block_def) if isinstance(block_def, list) else build_layer_or_network(block_def) for _ in range(count)]
        return nn.Sequential(*layers)

    # Organism-level architectures
    if op == "transformer-encoder-block":
        return _build_transformer_encoder_block(params)
    if op == "transformer-decoder-block":
        return _build_transformer_decoder_block(params)
    if op == "mlp":
        return _build_mlp(params)

    # Single layer as top-level network
    return build_layer_or_network(architecture)


def build_layer_or_network(spec: Any) -> nn.Module:
    """Build a single layer or recursively build a sub-network."""
    if isinstance(spec, list) and len(spec) >= 1:
        op = _normalize_op(str(spec[0]))
        if op in ("sequential", "parallel", "residual", "repeat",
                   "transformer-encoder-block", "transformer-decoder-block", "mlp"):
            return build_network(spec)
        return build_layer(spec)
    raise ValueError(f"Invalid spec: {spec}")


def build_layer(spec: Any) -> nn.Module:
    """
    Build a single layer from S-expression.

    Supports both positional: ["linear", 16, 64]
    and dict-style: ["conv2d", {"in-channels": 3, "out-channels": 64, "kernel": 3}]
    """
    op, params = _parse_layer_spec(spec)

    builder = LAYER_BUILDERS.get(op)
    if builder:
        return builder(params)

    raise ValueError(f"Unknown layer type: {op} (from spec: {spec})")


# ---------------------------------------------------------------------------
# Layer Builder Functions
# ---------------------------------------------------------------------------

def _build_linear(p):
    return nn.Linear(_p(p, "in", 0), _p(p, "out", 1), bias=_p(p, "bias", 2, True))

def _build_bilinear(p):
    return nn.Bilinear(_p(p, "in1", 0), _p(p, "in2", 1), _p(p, "out", 2))

def _build_embedding(p):
    return nn.Embedding(_p(p, "num", 0), _p(p, "dim", 1), padding_idx=_p(p, "padding-idx", 2))

def _build_conv1d(p):
    return nn.Conv1d(_p(p, "in-channels", 0), _p(p, "out-channels", 1), _p(p, "kernel", 2),
                     stride=_p(p, "stride", 3, 1), padding=_p(p, "padding", 4, 0))

def _build_conv2d(p):
    return nn.Conv2d(_p(p, "in-channels", 0), _p(p, "out-channels", 1), _p(p, "kernel", 2),
                     stride=_p(p, "stride", 3, 1), padding=_p(p, "padding", 4, 0))

def _build_conv3d(p):
    return nn.Conv3d(_p(p, "in-channels", 0), _p(p, "out-channels", 1), _p(p, "kernel", 2),
                     stride=_p(p, "stride", 3, 1), padding=_p(p, "padding", 4, 0))

def _build_conv_transpose2d(p):
    return nn.ConvTranspose2d(_p(p, "in-channels", 0), _p(p, "out-channels", 1), _p(p, "kernel", 2),
                              stride=_p(p, "stride", 3, 1), padding=_p(p, "padding", 4, 0),
                              output_padding=_p(p, "output-padding", 5, 0))

def _build_lstm(p):
    return nn.LSTM(_p(p, "input-size", 0), _p(p, "hidden-size", 1),
                   num_layers=_p(p, "num-layers", 2, 1),
                   bidirectional=_p(p, "bidirectional", 3, False),
                   dropout=_p(p, "dropout", 4, 0),
                   batch_first=_p(p, "batch-first", 5, True))

def _build_gru(p):
    return nn.GRU(_p(p, "input-size", 0), _p(p, "hidden-size", 1),
                  num_layers=_p(p, "num-layers", 2, 1),
                  bidirectional=_p(p, "bidirectional", 3, False),
                  dropout=_p(p, "dropout", 4, 0),
                  batch_first=_p(p, "batch-first", 5, True))

def _build_multi_head_attention(p):
    return nn.MultiheadAttention(_p(p, "embed-dim", 0), _p(p, "num-heads", 1),
                                  dropout=_p(p, "dropout", 2, 0), batch_first=True)

# Activations
def _build_relu(_): return nn.ReLU()
def _build_gelu(p): return nn.GELU(approximate=_p(p, "approximate", 0, "none"))
def _build_silu(_): return nn.SiLU()
def _build_tanh(_): return nn.Tanh()
def _build_sigmoid(_): return nn.Sigmoid()
def _build_softmax(p): return nn.Softmax(dim=_p(p, "dim", 0, -1))
def _build_leaky_relu(p): return nn.LeakyReLU(negative_slope=_p(p, "negative-slope", 0, 0.01))
def _build_elu(p): return nn.ELU(alpha=_p(p, "alpha", 0, 1.0))
def _build_mish(_): return nn.Mish()
def _build_swiglu(p): return SwiGLU(_p(p, "dim", 0), _p(p, "ff-dim", 1))

# Normalization
def _build_batch_norm(p): return nn.BatchNorm1d(_p(p, "num-features", 0), momentum=_p(p, "momentum", 1, 0.1), eps=_p(p, "eps", 2, 1e-5))
def _build_layer_norm(p): return nn.LayerNorm(_p(p, "normalized-shape", 0), eps=_p(p, "eps", 1, 1e-5))
def _build_group_norm(p): return nn.GroupNorm(_p(p, "num-groups", 0), _p(p, "num-channels", 1), eps=_p(p, "eps", 2, 1e-5))
def _build_instance_norm(p): return nn.InstanceNorm1d(_p(p, "num-features", 0))
def _build_rms_norm(p): return RMSNorm(_p(p, "dim", 0), eps=_p(p, "eps", 1, 1e-5))

# Regularization
def _build_dropout(p): return nn.Dropout(_p(p, "rate", 0, 0.5))
def _build_dropout2d(p): return nn.Dropout2d(_p(p, "rate", 0, 0.5))
def _build_alpha_dropout(p): return nn.AlphaDropout(_p(p, "rate", 0, 0.5))

# Pooling
def _build_max_pool2d(p): return nn.MaxPool2d(_p(p, "kernel", 0), stride=_p(p, "stride", 1), padding=_p(p, "padding", 2, 0))
def _build_avg_pool2d(p): return nn.AvgPool2d(_p(p, "kernel", 0), stride=_p(p, "stride", 1), padding=_p(p, "padding", 2, 0))
def _build_adaptive_avg_pool(p): return nn.AdaptiveAvgPool2d(_p(p, "output-size", 0))
def _build_global_mean_pool(_): return GlobalMeanPool()
def _build_global_max_pool(_): return GlobalMaxPool()

# Shape
def _build_flatten(p): return nn.Flatten(start_dim=_p(p, "start-dim", 0, 1), end_dim=_p(p, "end-dim", 1, -1))
def _build_reshape(p): return Reshape(_p(p, "shape", 0))
def _build_select_token(p): return SelectToken(_p(p, "index", 0, 0))
def _build_select_hidden(p): return SelectHidden(_p(p, "method", 0, "last"))

# Embedding / Position
def _build_positional_encoding(p): return PositionalEncoding(_p(p, "dim", 0), _p(p, "max-len", 1, 5000), _p(p, "type", 2, "sinusoidal"))
def _build_patch_embedding(p): return PatchEmbedding(_p(p, "image-size", 0), _p(p, "patch-size", 1), _p(p, "in-channels", 2, 3), _p(p, "embed-dim", 3))
def _build_prepend_cls_token(p): return PrependCLSToken(_p(p, "dim", 0))
def _build_token_type_embedding(p): return TokenTypeEmbedding(_p(p, "num", 0), _p(p, "dim", 1))
def _build_pooler(p): return Pooler(_p(p, "method", 0), _p(p, "dim", 1))


# ---------------------------------------------------------------------------
# Layer Registry
# ---------------------------------------------------------------------------

LAYER_BUILDERS: Dict[str, Any] = {
    # Core
    "linear": _build_linear,
    "bilinear": _build_bilinear,
    "embedding": _build_embedding,
    # Vision
    "conv1d": _build_conv1d,
    "conv2d": _build_conv2d,
    "conv3d": _build_conv3d,
    "conv-transpose2d": _build_conv_transpose2d,
    # Recurrent
    "lstm": _build_lstm,
    "gru": _build_gru,
    # Attention
    "multi-head-attention": _build_multi_head_attention,
    # Activations
    "relu": _build_relu,
    "gelu": _build_gelu,
    "silu": _build_silu,
    "tanh": _build_tanh,
    "sigmoid": _build_sigmoid,
    "softmax": _build_softmax,
    "leaky-relu": _build_leaky_relu,
    "elu": _build_elu,
    "mish": _build_mish,
    "swiglu": _build_swiglu,
    # Normalization
    "batch-norm": _build_batch_norm,
    "layer-norm": _build_layer_norm,
    "group-norm": _build_group_norm,
    "instance-norm": _build_instance_norm,
    "rms-norm": _build_rms_norm,
    # Regularization
    "dropout": _build_dropout,
    "dropout2d": _build_dropout2d,
    "alpha-dropout": _build_alpha_dropout,
    # Pooling
    "max-pool2d": _build_max_pool2d,
    "avg-pool2d": _build_avg_pool2d,
    "adaptive-avg-pool": _build_adaptive_avg_pool,
    "global-mean-pool": _build_global_mean_pool,
    "global-max-pool": _build_global_max_pool,
    # Shape
    "flatten": _build_flatten,
    "reshape": _build_reshape,
    "select-token": _build_select_token,
    "select-hidden": _build_select_hidden,
    # Embedding / Position
    "positional-encoding": _build_positional_encoding,
    "patch-embedding": _build_patch_embedding,
    "prepend-cls-token": _build_prepend_cls_token,
    "token-type-embedding": _build_token_type_embedding,
    "pooler": _build_pooler,

    # Legacy nn/ prefixed names (backward compat)
    "batchnorm": _build_batch_norm,      # nn/batchnorm (legacy)
    "batchnorm1d": _build_batch_norm,    # nn/batchnorm1d
    "layernorm": _build_layer_norm,      # nn/layernorm (legacy)
}


# ---------------------------------------------------------------------------
# Organism Builders
# ---------------------------------------------------------------------------

def _build_transformer_encoder_block(params) -> nn.Module:
    """Build a single transformer encoder block from params."""
    p = params if isinstance(params, dict) else {}
    if isinstance(params, list) and len(params) >= 4:
        p = {"embed-dim": params[0], "num-heads": params[1], "ff-dim": params[2], "dropout": params[3] if len(params) > 3 else 0.1}

    return TransformerEncoderBlock(
        embed_dim=p.get("embed-dim", 512),
        num_heads=p.get("num-heads", 8),
        ff_dim=p.get("ff-dim", 2048),
        dropout=p.get("dropout", 0.1),
        activation=p.get("activation", "relu"),
        norm=p.get("norm", "post"),
        norm_type=p.get("norm-type", "layer-norm"),
        causal_mask=p.get("causal-mask", False),
    )


def _build_transformer_decoder_block(params) -> nn.Module:
    """Build a single transformer decoder block from params."""
    p = params if isinstance(params, dict) else {}
    if isinstance(params, list) and len(params) >= 4:
        p = {"embed-dim": params[0], "num-heads": params[1], "ff-dim": params[2], "dropout": params[3] if len(params) > 3 else 0.1}

    return TransformerDecoderBlock(
        embed_dim=p.get("embed-dim", 512),
        num_heads=p.get("num-heads", 8),
        ff_dim=p.get("ff-dim", 2048),
        dropout=p.get("dropout", 0.1),
        cross_attention=p.get("cross-attention", False),
        activation=p.get("activation", "relu"),
        norm=p.get("norm", "post"),
        norm_type=p.get("norm-type", "layer-norm"),
    )


def _build_mlp(params) -> nn.Module:
    """Build MLP from dims list."""
    p = params if isinstance(params, dict) else {}
    if isinstance(params, list):
        p = {"dims": params[0] if params else []}

    dims = p.get("dims", [])
    activation = p.get("activation", "relu")
    dropout = p.get("dropout", 0)
    norm = p.get("norm")

    act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh, "leaky-relu": nn.LeakyReLU}
    act_cls = act_map.get(activation, nn.ReLU)

    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:  # No activation/norm/dropout on last layer
            if norm == "layer":
                layers.append(nn.LayerNorm(dims[i + 1]))
            elif norm == "batch":
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


def _build_organism_from_dict(architecture: dict) -> nn.Module:
    """Build organisms that use dict format (unet, conditional-unet)."""
    # Placeholder for complex organism construction
    raise NotImplementedError("Dict-form organisms (unet, conditional-unet) not yet implemented. Use sequential composition.")
