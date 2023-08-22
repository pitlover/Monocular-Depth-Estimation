from typing import Tuple, Optional
import math
import torch
import torch.nn as nn


class LimeConvBlock(nn.Module):

    def __init__(self,
                 in_ch: int,
                 mid_ch: int,
                 num_groups: int = 1,
                 act_layer=nn.GELU):
        super().__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, (1, 1), bias=False),
            nn.BatchNorm2d(mid_ch),
            act_layer()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, (3, 3), padding=(1, 1), groups=num_groups,
                      padding_mode="replicate", bias=False),
            nn.BatchNorm2d(mid_ch),
            act_layer()
        )
        # self.se = nn.Sequential(
        #     nn.Linear(mid_ch, mid_ch // 4),
        #     act_layer(),
        #     nn.Linear(mid_ch // 4, mid_ch),
        #     nn.Sigmoid(),
        # )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_ch, in_ch, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(in_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert c == self.in_ch

        identity = x
        x = self.conv1(x)
        x = self.conv2(x)

        # x_mean = torch.mean(x, dim=[2, 3])  # (b, 4d)
        # se = self.se(x_mean)
        # x = x * se.view(b, -1, 1, 1)  # (b, 4d, 1 , 1)

        x = self.conv3(x)
        x = x + identity
        return x


class LimeCrossAttention(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 enc_dim: int,
                 attn_drop_prob: float = 0.1,
                 drop_prob: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim

        self.norm = nn.LayerNorm(hidden_dim)
        self.enc_norm = nn.LayerNorm(enc_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(enc_dim, hidden_dim)
        self.v_proj = nn.Linear(enc_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_drop = nn.Dropout(attn_drop_prob, inplace=False)
        self.drop = nn.Dropout(drop_prob, inplace=False)

    def forward(self, hidden: torch.Tensor, enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s, d = hidden.shape
        _, enc_s, enc_d = enc.shape
        assert s == enc_s

        x = self.norm(hidden)
        enc = self.enc_norm(enc)

        q = self.q_proj(x)  # (b, s, d)
        k = self.k_proj(enc)  # (b, enc_s, enc_d)
        v = self.v_proj(enc)  # (b, enc_s, enc_d)

        attn = torch.matmul(k.transpose(-2, -1), q)  # (b, enc_d, d)
        attn *= math.sqrt(1 / s)
        attn = torch.softmax(attn, dim=-2)  # (b, enc_d, d)
        attn_drop = self.attn_drop(attn)

        out = torch.matmul(v, attn_drop)  # (b, enc_s, d)
        out = self.o_proj(out)

        out = self.drop(out)
        out = out + hidden

        return out, attn


class LimeLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 enc_dim: int,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1,
                 act_layer=nn.GELU):
        super().__init__()

        self.conv = LimeConvBlock(hidden_dim, hidden_dim, num_groups=1, act_layer=act_layer)
        self.attn = LimeCrossAttention(hidden_dim, enc_dim, attn_drop_prob, drop_prob)
        # self.norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                hidden: torch.Tensor,
                enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = hidden.shape
        # b, hw, enc_c = enc.shape

        hidden = self.conv(hidden)
        hidden = hidden.permute(0, 2, 3, 1).reshape(b, -1, c)  # (b, hw, c)
        hidden, attn = self.attn(hidden, enc)
        # hidden = self.norm(hidden)
        hidden = hidden.transpose(-2, -1).reshape(b, c, h, w)

        return hidden, attn
