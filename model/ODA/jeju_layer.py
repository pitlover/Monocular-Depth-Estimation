from typing import Tuple, Optional
import math
import torch
import torch.nn as nn


class JejuFeedForward(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_groups: int,
                 feedforward_dim: Optional[int] = None,
                 drop_prob: float = 0.1,
                 act_layer=nn.GELU):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        if feedforward_dim is None:
            feedforward_dim = 4 * hidden_dim
        self.feedforward_dim = feedforward_dim

        # self.norm = nn.LayerNorm(hidden_dim)
        # self.conv1 = nn.Sequential(
        #     nn.Linear(hidden_dim, feedforward_dim),
        #     act_layer(),
        #     nn.Dropout(drop_prob),
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden_dim, feedforward_dim, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(feedforward_dim),
            act_layer(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(feedforward_dim, feedforward_dim, kernel_size=(5, 5), stride=(1, 1),
                      padding=(2, 2), padding_mode="replicate", groups=num_groups, bias=False),
            nn.BatchNorm2d(feedforward_dim),
            act_layer(),
        )
        self.se = nn.Sequential(
            nn.Linear(feedforward_dim, feedforward_dim // 16),
            act_layer(),
            nn.Linear(feedforward_dim // 16, feedforward_dim),
            nn.Sigmoid(),
        )
        # self.conv3 = nn.Sequential(
        #     nn.Linear(feedforward_dim, hidden_dim),
        #     nn.Dropout(drop_prob),
        # )
        self.conv3 = nn.Sequential(
            nn.Conv2d(feedforward_dim, hidden_dim, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim)
        )

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        # size = (h, w)
        b, s, d = x.shape  # s = h * w
        h, w = size
        assert (s == h * w) and (d == self.hidden_dim)

        x = x.transpose(1, 2).reshape(b, -1, h, w)
        identity = x
        x = self.conv1(x)  # (b, s, 4d)
        x = self.conv2(x)  # (b, 4d, h, w)

        x_mean = torch.mean(x, dim=[2, 3])  # (b, 4d)
        se = self.se(x_mean)
        x = x * se.view(b, -1, 1, 1)  # (b, 4d, 1 , 1)

        x = self.conv3(x)  # (b, s, d)
        # x = x.view(b, -1, s).transpose(1, 2).contiguous()
        out = x + identity

        # out = self.norm(out)  # (b, s, d)
        return out


class JejuBlock(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 enc_dim: int,
                 aux_dim: int,
                 num_heads: int,
                 qk_proj_dim: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self.aux_dim = aux_dim
        if qk_proj_dim is None:
            qk_proj_dim = aux_dim
        self.qk_proj_dim = qk_proj_dim
        self.num_heads = num_heads

        if qk_proj_dim % num_heads != 0:
            raise ValueError("Hidden dim not multiple of num heads.")
        self.head_dim = qk_proj_dim // num_heads

        self.q1_proj = nn.Linear(aux_dim, qk_proj_dim)
        self.k1_proj = nn.Linear(hidden_dim + enc_dim, qk_proj_dim)
        self.v1_proj = nn.Linear(hidden_dim + enc_dim, aux_dim)
        self.o1_proj = nn.Linear(aux_dim, aux_dim)

        self.q2_proj = nn.Linear(hidden_dim, qk_proj_dim)
        self.k2_proj = nn.Linear(aux_dim, qk_proj_dim)
        self.v2_proj = nn.Linear(aux_dim, hidden_dim)
        self.o2_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_scale = math.sqrt(1.0 / self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop_prob, inplace=False)
        self.drop = nn.Dropout(drop_prob, inplace=False)

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        # self.aux_norm = nn.LayerNorm(aux_dim, eps=1e-5)
        # self.enc_norm = nn.LayerNorm(enc_dim, eps=1e-5)
        # self.inter_norm = nn.LayerNorm(aux_dim, eps=1e-5)

    def _reshape_3d(self, x: torch.Tensor) -> torch.Tensor:
        # (b, K, d) -> (b, K, nh, hd) -> (b, nh, K, hd)
        b, k, d = x.shape
        x = x.view(b, k, self.num_heads, -1).transpose(1, 2).contiguous()
        return x

    def forward(self,
                hidden: torch.Tensor,
                enc: torch.Tensor,
                aux: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Luna sub-block, post-norm.

        :param hidden:      (batch_size, HW, hidden_dim)
        :param enc:         (batch_size, HW, enc_dim)
        :param aux:         (batch_size, K, aux_dim)
        :return:            (batch_size, HW, hidden_dim)
                            (batch_size, K, aux_dim)
                            (batch_size, num_heads, HW, K), (batch_size, num_heads, K, HW)
        """
        b, _, hidden_dim = hidden.shape
        _, n_aux, aux_dim = aux.shape
        assert (hidden_dim == self.hidden_dim) and (aux_dim == self.aux_dim)

        # --------------------------------------------------------------- #
        # 1st multi-head attention

        # projection
        hidden_enc = torch.cat([hidden, enc], dim=-1)  # (b, HW, d_enc + d)
        q1 = self.q1_proj(aux)  # (b, K, d_aux) -> (b, K, d_aux)
        k1 = self.k1_proj(hidden_enc)  # (b, HW, d_enc + d) -> (b, HW, d_aux)
        v1 = self.v1_proj(hidden_enc)  # (b, HW, d_enc + d) -> (b, HW, d_aux)

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
        out1 = out1.transpose(1, 2).reshape(b, -1, aux_dim)

        # output projection
        out1 = self.o1_proj(out1)  # (b, K, d_aux) -> (b, K, d_aux)
        # out1_drop = self.drop(out1)

        aux_out = aux + out1
        # aux_out = aux + out1_drop
        # aux_out = self.aux_norm(aux_out)

        # --------------------------------------------------------------- #
        # 2nd multi-head attention

        # projection
        q2 = self.q2_proj(hidden)  # (b, HW, d)
        k2 = self.k2_proj(aux_out)  # (b, K, d)
        v2 = self.v2_proj(aux_out)  # (b, K, d)

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
        out2_drop = self.drop(out2)

        out = hidden + out2_drop
        out = self.norm(out)

        return out, aux_out, attn1, attn2


class JejuLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 enc_dim: int,
                 aux_dim: int,
                 num_heads: int, *,
                 qk_proj_dim: Optional[int] = None,
                 feedforward_dim: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1,
                 act_layer=nn.GELU):
        super().__init__()

        self.jeju_attn = JejuBlock(
            hidden_dim, enc_dim, aux_dim, num_heads, qk_proj_dim, attn_drop_prob, drop_prob
        )
        self.jeju_ff = JejuFeedForward(
            hidden_dim, num_heads, feedforward_dim, drop_prob, act_layer
        )

    def forward(self,
                hidden: torch.Tensor,
                enc: torch.Tensor,
                aux: torch.Tensor,
                size: Tuple[int, int],
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Luna layer = Luna + FF

        :param hidden:      (batch_size, HW, hidden_dim)
        :param enc:         (batch_size, HW, enc_dim)
        :param aux:         (batch_size, K, aux_dim)
        :param size:        (H, W)
        :return:            (batch_size, HW, hidden_dim)
                            (batch_size, K, aux_dim)
                            (batch_size, num_heads, HW, K), (batch_size, num_heads, K, HW)
        """
        hidden, aux, attn1, attn2 = self.jeju_attn(hidden, enc, aux)
        hidden = self.jeju_ff(hidden, size=size)

        return hidden, aux, attn1, attn2
