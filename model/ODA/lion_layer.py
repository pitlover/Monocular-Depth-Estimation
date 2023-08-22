from typing import Tuple, Optional
import math
import torch
import torch.nn as nn


class LionFeedForwardConv(nn.Module):

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

        self.norm = nn.LayerNorm(hidden_dim)

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
            nn.Linear(feedforward_dim, feedforward_dim // 4),
            act_layer(),
            nn.Linear(feedforward_dim // 4, feedforward_dim),
            nn.Sigmoid(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(feedforward_dim, hidden_dim, kernel_size=(1, 1), bias=True),
            # nn.BatchNorm2d(hidden_dim)
            nn.Dropout(drop_prob),
        )

    def forward(self, x: torch.Tensor, identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, h, w, ch = x.shape

        if identity is None:
            identity = x

        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.conv1(x)  # (b, 4d, h, w)
        x = self.conv2(x)  # (b, 4d, h, w)

        x_mean = torch.mean(x, dim=[2, 3])  # (b, 4d)
        se = self.se(x_mean)
        x = x * se.view(b, -1, 1, 1)  # (b, 4d, 1 , 1)

        x = self.conv3(x)  # (b, d, h, w)
        x = x.permute(0, 2, 3, 1).contiguous()

        out = x + identity
        return out


class LionFeedForward(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 feedforward_dim: Optional[int] = None,
                 drop_prob: float = 0.1,
                 act_layer=nn.GELU):
        super().__init__()

        self.hidden_dim = hidden_dim
        if feedforward_dim is None:
            feedforward_dim = 4 * hidden_dim
        self.feedforward_dim = feedforward_dim

        self.norm = nn.LayerNorm(hidden_dim)
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.Dropout(drop_prob),
            act_layer(),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(drop_prob),
        )

    def forward(self, x: torch.Tensor, identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, h, w, d = x.shape

        if identity is None:
            identity = x

        x = self.norm(x)
        x = self.block(x)

        out = x + identity
        return out


class LionUpscale(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), padding_mode="replicate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, ch, h, w = x.shape
        x = self.up(x)
        x = self.conv(x)
        return x


class LionReorder(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim
        # self.conv = nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=(1, 1), bias=False)
        self.conv = nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=(3, 3),
                              padding=(1, 1), padding_mode="replicate", bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, h, w, = x.shape

        x = x.view(b, 4, -1, h, w)
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x3 = x[:, 3]

        y = torch.zeros(b, d // 4, 2 * h, 2 * w, device=x.device, dtype=x.dtype)
        y[:, :, 0::2, 0::2] = x0
        y[:, :, 1::2, 0::2] = x1
        y[:, :, 0::2, 1::2] = x2
        y[:, :, 1::2, 1::2] = x3

        y = self.conv(y)
        return y


class LionSelfAttentionDimH(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 qk_proj_dim: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        if qk_proj_dim is None:
            qk_proj_dim = hidden_dim

        self.norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, qk_proj_dim)
        self.k_proj = nn.Linear(hidden_dim, qk_proj_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_drop = nn.Dropout(attn_drop_prob, inplace=False)
        self.drop = nn.Dropout(drop_prob, inplace=False)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, w, d = hidden.shape

        x = self.norm(hidden)
        q = self.q_proj(x)  # (b, h, w, d)
        k = self.k_proj(x)  # (b, h, w, d)
        v = self.v_proj(x)  # (b, h, w, d)

        # num_heads = h
        head_dim = w

        attn = torch.matmul(q.transpose(-2, -1), k)  # (b, h, d, d)
        attn *= math.sqrt(1 / head_dim)
        attn = torch.softmax(attn, dim=-2)  # (b, h, d, d)
        attn_drop = self.attn_drop(attn)

        out = torch.matmul(v, attn_drop)  # (b, h, w, d)
        out = self.o_proj(out)

        out = self.drop(out)
        out = out + hidden

        return out, attn


class LionSelfAttentionDimW(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 qk_proj_dim: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        if qk_proj_dim is None:
            qk_proj_dim = hidden_dim

        self.norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, qk_proj_dim)
        self.k_proj = nn.Linear(hidden_dim, qk_proj_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_drop = nn.Dropout(attn_drop_prob, inplace=False)
        self.drop = nn.Dropout(drop_prob, inplace=False)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, w, d = hidden.shape

        x = self.norm(hidden).transpose(1, 2).contiguous()
        q = self.q_proj(x)  # (b, w, h, d)
        k = self.k_proj(x)  # (b, w, h, d)
        v = self.v_proj(x)  # (b, w, h, d)

        # num_heads = w
        head_dim = h

        attn = torch.matmul(q.transpose(-2, -1), k)  # (b, w, d, d)
        attn *= math.sqrt(1 / head_dim)
        attn = torch.softmax(attn, dim=-2)  # (b, w, d, d)
        attn_drop = self.attn_drop(attn)

        out = torch.matmul(v, attn_drop)  # (b, w, h, d)
        out = self.o_proj(out)
        out = self.drop(out)
        out = out.transpose(1, 2).contiguous()  # (b, h, w, d)

        out = out + hidden
        return out, attn


class LionCrossAttentionDimH(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 enc_dim: int,
                 qk_proj_dim: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        if qk_proj_dim is None:
            qk_proj_dim = hidden_dim

        self.norm = nn.LayerNorm(hidden_dim)
        self.enc_norm = nn.LayerNorm(enc_dim)

        self.q_proj = nn.Linear(hidden_dim, qk_proj_dim)
        self.k_proj = nn.Linear(enc_dim, qk_proj_dim)
        self.v_proj = nn.Linear(enc_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_drop = nn.Dropout(attn_drop_prob, inplace=False)
        self.drop = nn.Dropout(drop_prob, inplace=False)

    def forward(self, hidden: torch.Tensor, enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, w, d = hidden.shape

        x = self.norm(hidden)
        enc = self.enc_norm(enc)
        q = self.q_proj(x)  # (b, h, w, d)
        k = self.k_proj(enc)  # (b, h, w, d)
        v = self.v_proj(enc)  # (b, h, w, d)

        # num_heads = h
        head_dim = w

        attn = torch.matmul(q.transpose(-2, -1), k)  # (b, h, d, d)
        attn *= math.sqrt(1 / head_dim)
        attn = torch.softmax(attn, dim=-2)  # (b, h, d, d)
        attn_drop = self.attn_drop(attn)

        out = torch.matmul(v, attn_drop)  # (b, h, w, d)
        out = self.o_proj(out)

        out = self.drop(out)
        out = out + hidden

        return out, attn


class LionCrossAttentionDimW(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 enc_dim: int,
                 qk_proj_dim: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        if qk_proj_dim is None:
            qk_proj_dim = hidden_dim

        self.norm = nn.LayerNorm(hidden_dim)
        self.enc_norm = nn.LayerNorm(enc_dim)

        self.q_proj = nn.Linear(hidden_dim, qk_proj_dim)
        self.k_proj = nn.Linear(enc_dim, qk_proj_dim)
        self.v_proj = nn.Linear(enc_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_drop = nn.Dropout(attn_drop_prob, inplace=False)
        self.drop = nn.Dropout(drop_prob, inplace=False)

    def forward(self, hidden: torch.Tensor, enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, w, d = hidden.shape

        x = self.norm(hidden)
        enc = self.enc_norm(enc)

        x = x.transpose(1, 2).contiguous()
        enc = enc.transpose(1, 2).contiguous()
        q = self.q_proj(x)  # (b, w, h, d)
        k = self.k_proj(enc)  # (b, w, h, d)
        v = self.v_proj(enc)  # (b, w, h, d)

        # num_heads = w
        head_dim = h

        attn = torch.matmul(q.transpose(-2, -1), k)  # (b, w, d, d)
        attn *= math.sqrt(1 / head_dim)
        attn = torch.softmax(attn, dim=-2)  # (b, w, d, d)
        attn_drop = self.attn_drop(attn)

        out = torch.matmul(v, attn_drop)  # (b, w, h, d)
        out = self.o_proj(out)
        out = self.drop(out)
        out = out.transpose(1, 2).contiguous()  # (b, h, w, d)

        out = out + hidden
        return out, attn


class LionLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 enc_dim: int,
                 qk_proj_dim: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1,
                 act_layer=nn.GELU,
                 last_block: bool = False):
        super().__init__()

        # self.enc_proj = nn.Linear(enc_dim, hidden_dim)

        self.attn_h = LionSelfAttentionDimH(hidden_dim, qk_proj_dim, attn_drop_prob, drop_prob)
        self.cross_attn_h = LionCrossAttentionDimH(hidden_dim, enc_dim, qk_proj_dim, attn_drop_prob, drop_prob)
        # self.feed_forward_h = LionFeedForward(hidden_dim, drop_prob=drop_prob, act_layer=act_layer)
        self.feed_forward_h = LionFeedForwardConv(hidden_dim, feedforward_dim=hidden_dim,
                                                  num_groups=1, act_layer=act_layer)

        self.attn_w = LionSelfAttentionDimW(hidden_dim, qk_proj_dim, attn_drop_prob, drop_prob)
        self.cross_attn_w = LionCrossAttentionDimW(hidden_dim, enc_dim, qk_proj_dim, attn_drop_prob, drop_prob)
        # self.feed_forward_w = LionFeedForward(hidden_dim, drop_prob=drop_prob, act_layer=act_layer)
        self.feed_forward_w = LionFeedForwardConv(hidden_dim, feedforward_dim=hidden_dim,
                                                  num_groups=1, act_layer=act_layer)

        # self.ln_attn = nn.LayerNorm(hidden_dim)
        self.upscale = LionReorder(hidden_dim)

        self.last_block = last_block
        if not last_block:
            self.out = nn.LayerNorm(hidden_dim // 2)
        else:
            self.out = nn.Sequential(
                nn.BatchNorm2d(hidden_dim // 2, eps=1e-5),
                act_layer()
            )

    def forward(self,
                hidden: torch.Tensor,
                enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # b, h, w, c = hidden.shape
        # enc = self.enc_proj(enc)

        hidden, attn = self.attn_h(hidden)
        hidden, cross_attn = self.cross_attn_h(hidden, enc)
        hidden = self.feed_forward_h(hidden)

        hidden, attn = self.attn_w(hidden)
        hidden, cross_attn = self.cross_attn_w(hidden, enc)
        hidden = self.feed_forward_w(hidden)

        # hidden = self.ln_attn(hidden)
        hidden = hidden.permute(0, 3, 1, 2).contiguous()  # (b, c, h, w)
        hidden = self.upscale(hidden)

        if not self.last_block:
            hidden = hidden.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)

        hidden = self.out(hidden)

        return hidden, attn, cross_attn


"""
class LionCrossAttentionHW(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 enc_dim: int,
                 qk_proj_dim: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        if qk_proj_dim is None:
            qk_proj_dim = hidden_dim

        self.norm = nn.LayerNorm(hidden_dim)
        self.enc_norm = nn.LayerNorm(enc_dim)
        self.inter_norm = nn.LayerNorm(hidden_dim)

        self.q_proj_h = nn.Linear(hidden_dim, qk_proj_dim)
        self.k_proj_h = nn.Linear(enc_dim, qk_proj_dim)
        self.v_proj_h = nn.Linear(enc_dim, hidden_dim)

        self.q_proj_w = nn.Linear(hidden_dim, qk_proj_dim)
        self.k_proj_w = nn.Linear(enc_dim, qk_proj_dim)
        self.v_proj_w = nn.Linear(enc_dim, hidden_dim)

        self.o_proj_h = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj_w = nn.Linear(hidden_dim, hidden_dim)

        self.attn_drop = nn.Dropout(attn_drop_prob, inplace=False)
        self.drop = nn.Dropout(drop_prob, inplace=False)

    def forward(self, hidden: torch.Tensor, enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, w, d = hidden.shape
        assert d == self.hidden_dim

        x = self.norm(hidden)
        enc = self.enc_norm(enc)

        # height
        q = self.q_proj_h(x)  # (b, h, w, d)
        k = self.k_proj_h(enc)  # (b, h, w, d)
        v = self.v_proj_h(enc)  # (b, h, w, d)

        # num_heads = h
        head_dim = d

        attn = torch.matmul(q, k.transpose(-2, -1))  # (b, h, w, w)
        attn *= math.sqrt(1 / head_dim)
        attn = torch.softmax(attn, dim=-1)  # (b, h, w, w)
        attn_drop = self.attn_drop(attn)

        out = torch.matmul(attn_drop, v)  # (b, h, w, d)
        out = self.o_proj_h(out)
        x = self.inter_norm(out)

        # width
        q = self.q_proj_w(x).transpose(1, 2).contiguous()  # (b, h, w, d) -> (b, w, h, d)
        k = self.k_proj_w(enc).transpose(1, 2).contiguous()  # (b, h, w, d) -> (b, w, h, d)
        v = self.v_proj_w(enc).transpose(1, 2).contiguous()  # (b, h, w, d) -> (b, w, h, d)

        # num_heads = w
        head_dim = d

        attn = torch.matmul(q, k.transpose(-2, -1))  # (b, w, h, h)
        attn *= math.sqrt(1 / head_dim)
        attn = torch.softmax(attn, dim=-1)  # (b, w, h, h)
        attn_drop = self.attn_drop(attn)

        out = torch.matmul(attn_drop, v)  # (b, w, h, d)
        out = self.o_proj_w(out)
        out = out.transpose(1, 2).contiguous()

        out = self.drop(out)
        out = out + hidden

        return out, attn
"""
