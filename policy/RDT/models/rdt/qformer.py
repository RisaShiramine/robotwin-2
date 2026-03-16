"""Language-Guided Q-Former for RDT.

Image tokens (Q) attend to language tokens (K/V) in each block, so that
visual features are refined with language context before being fed into RDT.

Architecture per block
----------------------
  1. Self-attention  among image tokens   (image understands global context)
  2. Cross-attention img → lang           (image looks up task-relevant info)
  3. Feed-Forward Network

Output shape == input img_tokens shape, so this module is a drop-in refinement
step between the img_adaptor and the RDT Transformer.
"""

import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, RmsNorm

from models.rdt.blocks import CrossAttention


class QFormerBlock(nn.Module):
    """One Q-Former block.

    Args:
        hidden_size: token feature dimension (must equal RDT hidden_size).
        num_heads:   number of attention heads.
    """

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        # --- self-attention ---
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)
        self.self_attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
        )

        # --- cross-attention: img (Q) ← lang (K, V) ---
        self.norm2 = RmsNorm(hidden_size, eps=1e-6)
        self.cross_attn = CrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
        )

        # --- FFN ---
        self.norm3 = RmsNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(
        self,
        img: torch.Tensor,
        lang: torch.Tensor,
        lang_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img:       [B, N_img,  D]  image tokens
            lang:      [B, N_lang, D]  language tokens
            lang_mask: [B, N_lang]     bool mask (True = valid token)

        Returns:
            img: [B, N_img, D]  language-grounded image tokens
        """
        # 1. self-attention
        img = img + self.self_attn(self.norm1(img))
        # 2. cross-attention: image attends to language
        img = img + self.cross_attn(self.norm2(img), lang, lang_mask)
        # 3. FFN
        img = img + self.ffn(self.norm3(img))
        return img


class LanguageGuidedQFormer(nn.Module):
    """Stacks multiple QFormerBlocks to refine image tokens with language context.

    Args:
        hidden_size: feature dimension (must equal RDT hidden_size).
        num_heads:   number of attention heads.
        num_layers:  number of stacked QFormerBlocks.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [QFormerBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )

    def forward(
        self,
        img_tokens: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img_tokens:  [B, N_img,  hidden_size]  adapted image tokens
            lang_tokens: [B, N_lang, hidden_size]  adapted language tokens
            lang_mask:   [B, N_lang]               bool mask (True = valid)

        Returns:
            [B, N_img, hidden_size]  language-grounded image tokens
        """
        for block in self.blocks:
            img_tokens = block(img_tokens, lang_tokens, lang_mask)
        return img_tokens
