"""Self-contained decoder-only transformer for the retok experiment.

Kept independent of other experiments' model code so the experiment extracts
cleanly (see /extract-release). Standard pre-norm decoder: learned positional
embeddings, GELU MLP, tied input/output embeddings. ``forward_with_residuals``
exposes the per-layer residual stream for the probe / logit-lens analysis.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from retok.config import RetokModelConfig


class _Attention(nn.Module):
    def __init__(self, config: RetokModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.wq = nn.Linear(config.dim, config.dim, bias=False)
        self.wk = nn.Linear(config.dim, config.dim, bias=False)
        self.wv = nn.Linear(config.dim, config.dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        q = self.wq(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.resid_dropout(self.wo(out))


class _MLP(nn.Module):
    def __init__(self, config: RetokModelConfig) -> None:
        super().__init__()
        self.w_up = nn.Linear(config.dim, config.intermediate_dim, bias=False)
        self.w_down = nn.Linear(config.intermediate_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_down(F.gelu(self.w_up(x))))


class _Block(nn.Module):
    def __init__(self, config: RetokModelConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.dim, eps=config.norm_eps)
        self.attn = _Attention(config)
        self.mlp_norm = nn.LayerNorm(config.dim, eps=config.norm_eps)
        self.mlp = _MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class RetokTransformer(nn.Module):
    """Decoder-only LM with tied embeddings and learned positional embeddings."""

    def __init__(self, config: RetokModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.dim)
        self.layers = nn.ModuleList(_Block(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.dim, eps=config.norm_eps)
        # Parallel one-step readout: D per-digit heads applied at the "=" position
        # to predict all answer digits at once (the matched no-CoT control). Each
        # head is only base-way, so this measures one-step arithmetic capability
        # without the base**D merged-token retrieval bottleneck.
        self.one_step_readout = nn.Linear(config.dim, config.n_digits * config.base)
        if config.init_std is not None:
            self._init_weights(config.init_std)

    def _init_weights(self, std: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def _embed(self, input_ids: Tensor) -> Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return self.tok_emb(input_ids) + self.pos_emb(positions).unsqueeze(0)

    def get_logits(self, hidden: Tensor) -> Tensor:
        return F.linear(hidden, self.tok_emb.weight)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self._embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.get_logits(self.final_norm(x))

    def forward_with_residuals(self, input_ids: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Return ``(logits, residuals)`` where ``residuals[i]`` is the stream
        after layer ``i`` (post-block, pre-final-norm)."""
        x = self._embed(input_ids)
        residuals: list[Tensor] = []
        for layer in self.layers:
            x = layer(x)
            residuals.append(x)
        return self.get_logits(self.final_norm(x)), residuals

    def one_step_logits(self, input_ids: Tensor, eq_position: int) -> Tensor:
        """Predict all D answer digits in parallel from the ``=`` position.

        Returns ``(B, n_digits, base)`` logits — the matched no-CoT control.
        The whole answer must be produced from a single position's hidden state,
        so accuracy on high-order digits (which need long carry propagation) is
        bounded by model depth, not by any retrieval vocabulary.
        """
        x = self._embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        h = self.final_norm(x)[:, eq_position, :]  # (B, dim)
        logits = self.one_step_readout(h)  # (B, n_digits*base)
        return logits.view(-1, self.config.n_digits, self.config.base)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: RetokModelConfig) -> RetokTransformer:
    return RetokTransformer(config)
