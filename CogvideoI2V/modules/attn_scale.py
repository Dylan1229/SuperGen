"""Attention logit scaling for high-resolution tiled denoising.

Why
---
Softmax attention over N keys has entropy that grows with N. At a larger canvas
each query attends over more tokens, the distribution flattens, and the output
moves toward an average of more values -- which reads as softness. Both
high-resolution competitors correct for it, and neither needed retraining to do so:

  CineScale (diffsynth/models/wan_video_dit.py:22-33, and duplicated in
  distributed/xdit_context_parallel.py:120-131):
      attention_scale = 1/sqrt(d) * log(45*80*coef, 45*80)
  with coef 1.5 or 2.0 selected by token count. Self-attention only
  (`if q.shape == v.shape`). Effect: +4.95% at 3K, +8.46% at 4K.

  FreeSwim (nocache/attention_processor_.py:160):
      attention_scale = sqrt(log(30*52*2, 30*52) / d)
  Effect: +4.61%. Applied to both of its attention paths.

Both amount to multiplying the logits by log_{T}(T*coef) = 1 + log(coef)/log(T),
where T is the reference token count. Raising the logit scale sharpens the softmax,
restoring peakiness lost to the larger key set.

Caveat specific to us
---------------------
Our tiles carry the SAME token count the model trained at (a 90x160 latent window
is exactly the 720p sequence length), so per-tile entropy dilution is much weaker
here than in CineScale/FreeSwim, which run attention over the full enlarged canvas.
This is therefore expected to be a smaller win for us than for them -- worth
measuring, not worth assuming. Default coef=1.0 leaves behaviour untouched.

A bug worth not copying: CineScale's two implementations disagree on the token
threshold (`q.shape[1]/21` vs `q.shape[1]/21*8`), which only agree when
world_size==8. We take the coefficient explicitly instead of inferring it.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Token count of one tile at the model's native resolution: (90/2)*(160/2) = 3600,
# the same 45*80 CineScale references.
REFERENCE_TOKENS = 45 * 80


def logit_scale(head_dim: int, coef: float = 1.0,
                reference_tokens: int = REFERENCE_TOKENS) -> float:
    """CineScale's scale: 1/sqrt(d) * log_{T}(T*coef). coef=1.0 -> stock 1/sqrt(d)."""
    base = 1.0 / math.sqrt(head_dim)
    if coef == 1.0:
        return base
    return base * math.log(reference_tokens * coef, reference_tokens)


class ScaledCogVideoXAttnProcessor:
    """CogVideoXAttnProcessor2_0 with an overridable attention logit scale.

    Transcribed from diffusers' processor so the only difference is the `scale=`
    argument to `scaled_dot_product_attention`. Cross-attention is left alone
    (CineScale also restricts the change to self-attention).
    """

    def __init__(self, coef: float = 1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("ScaledCogVideoXAttnProcessor requires PyTorch >= 2.0")
        self.coef = coef

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(
                query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(
                    key[:, :, text_seq_length:], image_rotary_emb)

        # The one substantive change: sharpen the softmax to counter the entropy
        # dilution that comes with attending over a larger key set.
        scale = logit_scale(head_dim, self.coef)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0,
            is_causal=False, scale=scale,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1)
        return hidden_states, encoder_hidden_states


def install_attention_scaling(transformer, coef: float = 1.0) -> int:
    """Swap in the scaled processor on every block. Returns how many were replaced."""
    if coef == 1.0:
        return 0
    n = 0
    for block in transformer.transformer_blocks:
        block.attn1.processor = ScaledCogVideoXAttnProcessor(coef)
        n += 1
    head_dim = transformer.config.attention_head_dim
    logger.info(
        f"Attention logit scaling on {n} blocks: coef={coef}, "
        f"scale {1/math.sqrt(head_dim):.6f} -> {logit_scale(head_dim, coef):.6f} "
        f"(+{(logit_scale(head_dim, coef)/(1/math.sqrt(head_dim)) - 1)*100:.2f}%)"
    )
    return n
