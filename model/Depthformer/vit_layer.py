from typing import Optional, Tuple
import torch
import torch.nn as nn

from .self_attention import SelfAttentionBlock
from .feed_forward import FeedForwardBlock


class ViTLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 key_query_dim : int,
                 num_heads: int, *,
                 num_repeat: int = 1,
                 feedforward_dim: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1,
                 act_layer=nn.GELU):
        super().__init__()

        if num_repeat < 1:
            raise ValueError("num_repeat is less than 1.")
        self.num_repeat = num_repeat

        self.self_attn = SelfAttentionBlock(
            hidden_dim, key_query_dim, num_heads, attn_drop_prob, drop_prob
        )
        self.feed_forward = FeedForwardBlock(
            hidden_dim, feedforward_dim, drop_prob, act_layer
        )

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """vit layer = SA + FF

        :param hidden:      (batch_size, num_patches, hidden_dim)
        :return:            (batch_size, num_patches, hidden_dim)
                            (batch_size, num_heads, num_patches, num_patches)
        """
        attn_weight = None
        for _ in range(self.num_repeat):
            hidden, attn_weight = self.self_attn(hidden)
            hidden = self.feed_forward(hidden)
        return hidden, attn_weight

