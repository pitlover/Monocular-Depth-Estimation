from typing import Tuple, Optional
import math
import torch
import torch.nn as nn

from .feed_forward import PostNormFeedForwardBlock, FeedForwardBlock
from .layer_utils import ResConvBNBlock


class LunaBlock(nn.Module):
    """Luna: Linear Unified Nested Attention
    https://arxiv.org/pdf/2106.01540.pdf
    """

    def __init__(self,
                 hidden_dim: int,
                 aux_dim: int,
                 qk_proj_dim: int,
                 num_heads: int,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.aux_dim = aux_dim
        self.qk_proj_dim = qk_proj_dim
        self.num_heads = num_heads

        if hidden_dim % num_heads != 0:
            raise ValueError("Hidden dim not multiple of num heads.")
        self.head_dim = hidden_dim // num_heads

        self.q1_proj = nn.Linear(aux_dim, qk_proj_dim)
        self.k1_proj = nn.Linear(hidden_dim, qk_proj_dim)
        self.v1_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o1_proj = nn.Linear(hidden_dim, aux_dim)

        self.q2_proj = nn.Linear(hidden_dim, qk_proj_dim)
        self.k2_proj = nn.Linear(aux_dim, qk_proj_dim)
        self.v2_proj = nn.Linear(aux_dim, hidden_dim)
        self.o2_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_scale = math.sqrt(1.0 / self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop_prob, inplace=False)
        self.drop = nn.Dropout(drop_prob, inplace=False)

        self.aux_norm = nn.LayerNorm(aux_dim, eps=1e-5)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-5)

    def _reshape_3d(self, x: torch.Tensor) -> torch.Tensor:
        # (b, K, d) -> (b, K, nh, hd) -> (b, nh, K, hd)
        b, k, d = x.shape
        x = x.view(b, k, self.num_heads, -1).transpose(1, 2).contiguous()
        return x

    def forward(self,
                hidden: torch.Tensor,
                aux: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Luna sub-block, post-norm.

        :param hidden:      (batch_size, HW, hidden_dim)
        :param aux:         (batch_size, K, aux_dim)
        :return:            (batch_size, HW, hidden_dim)
                            (batch_size, K, aux_dim)
                            (batch_size, num_heads, HW, K), (batch_size, num_heads, K, HW)
        """
        b, _, hidden_dim = hidden.shape
        assert hidden_dim == self.hidden_dim

        # --------------------------------------------------------------- #
        # 1st multi-head attention

        # projection
        q1 = self.q1_proj(aux)  # (b, K, d)
        k1 = self.k1_proj(hidden)  # (b, HW, d)
        v1 = self.v1_proj(hidden)  # (b, HW, d)

        # reshape
        q1 = self._reshape_3d(q1)  # (b, nh, K, hd)
        k1 = self._reshape_3d(k1)  # (b, nh, HW, hd)
        v1 = self._reshape_3d(v1)  # (b, nh, HW, hd)

        # scaled dot-product
        # (b, nh, K, hd) x (b, nh, hd, HW) = (b, nh, K, HW)
        attn1 = torch.matmul(q1, k1.transpose(-2, -1))
        attn1 *= self.attn_scale
        attn1 = torch.softmax(attn1, dim=-1)  # (b, nh, K, HW)
        attn1_drop = self.attn_drop(attn1)

        # value multiply
        out1 = torch.matmul(attn1_drop, v1)  # (b, nh, K, HW) x (b, nh, HW, nd) = (b, nh, K, nd)
        out1 = out1.transpose(1, 2).reshape(b, -1, hidden_dim)

        # output projection
        out1 = self.o1_proj(out1)  # (b, K, d)
        out1 = self.drop(out1)

        aux_out = self.aux_norm(aux + out1)

        # --------------------------------------------------------------- #
        # 2nd multi-head attention

        # projection
        q2 = self.q2_proj(hidden)  # (b, HW, d)
        k2 = self.k2_proj(out1)  # (b, K, d)
        v2 = self.v2_proj(out1)  # (b, K, d)

        # reshape
        q2 = self._reshape_3d(q2)  # (b, nh, HW, hd)
        k2 = self._reshape_3d(k2)  # (b, nh, K, hd)
        v2 = self._reshape_3d(v2)  # (b, nh, K, hd)

        # scaled dot-product
        # (b, nh, HW, hd) x (b, nh, hd, K) = (b, nh, HW, K)
        attn2 = torch.matmul(q2, k2.transpose(-2, -1))
        attn2 *= self.attn_scale
        attn2 = torch.softmax(attn2, dim=-1)  # (b, nh, HW, K)
        attn2_drop = self.attn_drop(attn2)

        # value multiply
        out2 = torch.matmul(attn2_drop, v2)  # (b, nh, HW, K) x (b, nh, K, nd) = (b, nh, HW, nd)
        out2 = out2.transpose(1, 2).reshape(b, -1, hidden_dim)  # (b, HW, d)

        # output projection
        out2 = self.o2_proj(out2)  # (b, HW, d)
        out2 = self.drop(out2)

        out = self.norm(hidden + out2)

        return out, aux_out, attn1, attn2


class PreNormLunaBlock(nn.Module):
    """Luna: Linear Unified Nested Attention
    https://arxiv.org/pdf/2106.01540.pdf
    """

    def __init__(self,
                 hidden_dim: int,
                 aux_dim: int,
                 qk_proj_dim: int,
                 num_heads: int,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.aux_dim = aux_dim
        self.qk_proj_dim = qk_proj_dim
        self.num_heads = num_heads

        if hidden_dim % num_heads != 0:
            raise ValueError("Hidden dim not multiple of num heads.")
        self.head_dim = hidden_dim // num_heads

        self.aux_norm = nn.LayerNorm(aux_dim, eps=1e-5)
        self.inter_norm = nn.LayerNorm(aux_dim, eps=1e-5)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-5)

        self.q1_proj = nn.Linear(aux_dim, qk_proj_dim)
        self.k1_proj = nn.Linear(hidden_dim, qk_proj_dim)
        self.v1_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o1_proj = nn.Linear(hidden_dim, aux_dim)

        self.q2_proj = nn.Linear(hidden_dim, qk_proj_dim)
        self.k2_proj = nn.Linear(aux_dim, qk_proj_dim)
        self.v2_proj = nn.Linear(aux_dim, hidden_dim)
        self.o2_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_scale = math.sqrt(1.0 / self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop_prob, inplace=False)
        self.drop = nn.Dropout(drop_prob, inplace=False)

    def _reshape_3d(self, x: torch.Tensor) -> torch.Tensor:
        # (b, K, d) -> (b, K, nh, hd) -> (b, nh, K, hd)
        b, k, d = x.shape
        x = x.view(b, k, self.num_heads, -1).transpose(1, 2).contiguous()
        return x

    def forward(self,
                hidden: torch.Tensor,
                aux: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Luna sub-block, post-norm.

        :param hidden:      (batch_size, HW, hidden_dim)
        :param aux:         (batch_size, K, aux_dim)
        :return:            (batch_size, HW, hidden_dim)
                            (batch_size, K, aux_dim)
                            (batch_size, num_heads, HW, K), (batch_size, num_heads, K, HW)
        """
        b, _, hidden_dim = hidden.shape
        assert hidden_dim == self.hidden_dim

        # --------------------------------------------------------------- #
        # 1st multi-head attention
        aux_n = self.aux_norm(aux)
        hidden_n = self.norm(hidden)

        # projection
        q1 = self.q1_proj(aux_n)  # (b, K, d)
        k1 = self.k1_proj(hidden_n)  # (b, HW, d)
        v1 = self.v1_proj(hidden_n)  # (b, HW, d)

        # reshape
        q1 = self._reshape_3d(q1)  # (b, nh, K, hd)
        k1 = self._reshape_3d(k1)  # (b, nh, HW, hd)
        v1 = self._reshape_3d(v1)  # (b, nh, HW, hd)

        # scaled dot-product
        # (b, nh, K, hd) x (b, nh, hd, HW) = (b, nh, K, HW)
        attn1 = torch.matmul(q1, k1.transpose(-2, -1))
        attn1 *= self.attn_scale
        attn1 = torch.softmax(attn1, dim=-1)  # (b, nh, K, HW)
        attn1_drop = self.attn_drop(attn1)

        # value multiply
        out1 = torch.matmul(attn1_drop, v1)  # (b, nh, K, HW) x (b, nh, HW, nd) = (b, nh, K, nd)
        out1 = out1.transpose(1, 2).reshape(b, -1, hidden_dim)

        # output projection
        out1 = self.o1_proj(out1)  # (b, K, d)
        out1 = self.drop(out1)

        aux_out = aux + out1

        # --------------------------------------------------------------- #
        # 2nd multi-head attention
        out_n = self.inter_norm(out1)

        # projection
        q2 = self.q2_proj(hidden_n)  # (b, HW, d)
        k2 = self.k2_proj(out_n)  # (b, K, d)
        v2 = self.v2_proj(out_n)  # (b, K, d)

        # reshape
        q2 = self._reshape_3d(q2)  # (b, nh, HW, hd)
        k2 = self._reshape_3d(k2)  # (b, nh, K, hd)
        v2 = self._reshape_3d(v2)  # (b, nh, K, hd)

        # scaled dot-product
        # (b, nh, HW, hd) x (b, nh, hd, K) = (b, nh, HW, K)
        attn2 = torch.matmul(q2, k2.transpose(-2, -1))
        attn2 *= self.attn_scale
        attn2 = torch.softmax(attn2, dim=-1)  # (b, nh, HW, K)
        attn2_drop = self.attn_drop(attn2)

        # value multiply
        out2 = torch.matmul(attn2_drop, v2)  # (b, nh, HW, K) x (b, nh, K, nd) = (b, nh, HW, nd)
        out2 = out2.transpose(1, 2).reshape(b, -1, hidden_dim)  # (b, HW, d)

        # output projection
        out2 = self.o2_proj(out2)  # (b, HW, d)
        out2 = self.drop(out2)

        out = hidden + out2

        return out, aux_out, attn1, attn2


class LunaLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 aux_dim: int,
                 qk_proj_dim: int,
                 num_heads: int, *,
                 feedforward_dim: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1,
                 act_layer=nn.GELU):
        super().__init__()

        self.luna_attn = LunaBlock(
            hidden_dim, aux_dim, qk_proj_dim, num_heads, attn_drop_prob, drop_prob
        )
        self.feed_forward = PostNormFeedForwardBlock(
            hidden_dim, feedforward_dim, drop_prob, act_layer, add_weight=1.0
        )

    def forward(self,
                hidden: torch.Tensor,
                aux: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Luna layer = Luna + FF

        :param hidden:      (batch_size, hidden_dim, H, W)
        :param aux:         (batch_size, K, aux_dim)
        :return:            (batch_size, hidden_dim, H, W)
                            (batch_size, K, aux_dim)
                            (batch_size, num_heads, HW, K), (batch_size, num_heads, K, HW)
        """
        b, d, h, w = hidden.shape
        hidden = hidden.view(b, d, h * w).transpose(1, 2).contiguous()  # (b, HW, d)

        hidden, aux, attn1, attn2 = self.luna_attn(hidden, aux)
        hidden = self.feed_forward(hidden)

        hidden = hidden.transpose(1, 2).reshape(b, d, h, w)  # (b, d, H, W)

        return hidden, aux, attn1, attn2


class PreNormLunaLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 aux_dim: int,
                 qk_proj_dim: int,
                 num_heads: int, *,
                 feedforward_dim: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1,
                 act_layer=nn.GELU):
        super().__init__()

        self.luna_attn = PreNormLunaBlock(
            hidden_dim, aux_dim, qk_proj_dim, num_heads, attn_drop_prob, drop_prob
        )
        self.feed_forward = FeedForwardBlock(
            hidden_dim, feedforward_dim, drop_prob, act_layer, add_weight=1.0
        )

    def forward(self,
                hidden: torch.Tensor,
                aux: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Luna layer = Luna + FF

        :param hidden:      (batch_size, hidden_dim, H, W)
        :param aux:         (batch_size, K, aux_dim)
        :return:            (batch_size, hidden_dim, H, W)
                            (batch_size, K, aux_dim)
                            (batch_size, num_heads, HW, K), (batch_size, num_heads, K, HW)
        """
        b, d, h, w = hidden.shape
        hidden = hidden.view(b, d, h * w).transpose(1, 2).contiguous()  # (b, HW, d)

        hidden, aux, attn1, attn2 = self.luna_attn(hidden, aux)
        hidden = self.feed_forward(hidden)

        hidden = hidden.transpose(1, 2).reshape(b, d, h, w)  # (b, d, H, W)

        return hidden, aux, attn1, attn2


class LunaConvLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 aux_dim: int,
                 qk_proj_dim: int,
                 num_heads: int, *,
                 feedforward_dim: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1,
                 act_layer=nn.GELU):
        super().__init__()

        self.luna_attn = LunaBlock(
            hidden_dim, aux_dim, qk_proj_dim, num_heads, attn_drop_prob, drop_prob
        )
        self.conv = ResConvBNBlock(
            hidden_dim, hidden_dim, kernel_size=3, num_layers=2, act_layer=act_layer
        )

    def forward(self,
                hidden: torch.Tensor,
                aux: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Luna layer = Luna + FF

        :param hidden:      (batch_size, hidden_dim, H, W)
        :param aux:         (batch_size, K, aux_dim)
        :return:            (batch_size, hidden_dim, H, W)
                            (batch_size, K, aux_dim)
                            (batch_size, num_heads, HW, K), (batch_size, num_heads, K, HW)
        """
        b, d, h, w = hidden.shape
        hidden = hidden.view(b, d, h * w).transpose(1, 2).contiguous()  # (b, HW, d)

        hidden, aux, attn1, attn2 = self.luna_attn(hidden, aux)

        hidden = hidden.transpose(1, 2).reshape(b, d, h, w)  # (b, d, H, W)
        hidden = self.conv(hidden)

        return hidden, aux, attn1, attn2


class LunaHalfBlock(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 aux_dim: int,
                 qk_proj_dim: int,
                 num_heads: int,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.aux_dim = aux_dim
        self.qk_proj_dim = qk_proj_dim
        self.num_heads = num_heads

        if hidden_dim % num_heads != 0:
            raise ValueError("Hidden dim not multiple of num heads.")
        self.head_dim = hidden_dim // num_heads

        self.q1_proj = nn.Linear(aux_dim, qk_proj_dim)
        self.k1_proj = nn.Linear(hidden_dim, qk_proj_dim)
        self.v1_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o1_proj = nn.Linear(hidden_dim, aux_dim)

        self.attn_scale = math.sqrt(1.0 / self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop_prob, inplace=False)
        self.drop = nn.Dropout(drop_prob, inplace=False)

        self.aux_norm = nn.LayerNorm(aux_dim, eps=1e-5)

    def _reshape_3d(self, x: torch.Tensor) -> torch.Tensor:
        # (b, K, d) -> (b, K, nh, hd) -> (b, nh, K, hd)
        b, k, d = x.shape
        x = x.view(b, k, self.num_heads, -1).transpose(1, 2).contiguous()
        return x

    def forward(self,
                hidden: torch.Tensor,
                aux: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Luna sub-block, post-norm.

        :param hidden:      (batch_size, HW, hidden_dim)
        :param aux:         (batch_size, K, aux_dim)
        :return:            (batch_size, HW, hidden_dim)
                            (batch_size, K, aux_dim)
                            (batch_size, num_heads, HW, K), (batch_size, num_heads, K, HW)
        """
        b, hidden_dim, h, w = hidden.shape
        hidden = hidden.view(b, hidden_dim, h * w).transpose(1, 2).contiguous()  # (b, HW, d)
        assert hidden_dim == self.hidden_dim

        # --------------------------------------------------------------- #
        # 1st multi-head attention

        # projection
        q1 = self.q1_proj(aux)  # (b, K, d)
        k1 = self.k1_proj(hidden)  # (b, HW, d)
        v1 = self.v1_proj(hidden)  # (b, HW, d)

        # reshape
        q1 = self._reshape_3d(q1)  # (b, nh, K, hd)
        k1 = self._reshape_3d(k1)  # (b, nh, HW, hd)
        v1 = self._reshape_3d(v1)  # (b, nh, HW, hd)

        # scaled dot-product
        # (b, nh, K, hd) x (b, nh, hd, HW) = (b, nh, K, HW)
        attn1 = torch.matmul(q1, k1.transpose(-2, -1))
        attn1 *= self.attn_scale
        attn1 = torch.softmax(attn1, dim=-1)  # (b, nh, K, HW)
        attn1_drop = self.attn_drop(attn1)

        # value multiply
        out1 = torch.matmul(attn1_drop, v1)  # (b, nh, K, HW) x (b, nh, HW, nd) = (b, nh, K, nd)
        out1 = out1.transpose(1, 2).reshape(b, -1, hidden_dim)

        # output projection
        out1 = self.o1_proj(out1)  # (b, K, d)
        out1 = self.drop(out1)

        aux_out = self.aux_norm(aux + out1)

        return aux_out, attn1
