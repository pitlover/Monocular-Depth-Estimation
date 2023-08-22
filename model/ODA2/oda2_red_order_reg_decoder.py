from typing import Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .oda2_layer_utils import ConvBN, _CONV_PADDING_MODE


class PreNormFF(nn.Module):

    def __init__(self,
                 in_dims: int,
                 drop_prob: float = 0.0,
                 feedforward_dims: Optional[int] = None,
                 act_layer=nn.GELU):
        super().__init__()
        if feedforward_dims is None:
            feedforward_dims = 4 * in_dims
        self.in_dims = in_dims

        self.norm = nn.LayerNorm(in_dims)
        self.lin1 = nn.Linear(in_dims, feedforward_dims)
        self.lin2 = nn.Linear(feedforward_dims, in_dims)
        self.drop = nn.Dropout(drop_prob)
        self.act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.norm(x)
        x = self.lin1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.lin2(x)
        x = self.drop(x)

        x = x + identity
        return x


class PreNormDWConvFF(nn.Module):

    def __init__(self,
                 in_dims: int,
                 drop_prob: float = 0.0,
                 feedforward_dims: Optional[int] = None,
                 kernel_size: int = 5,
                 act_layer=nn.GELU):
        super().__init__()
        if feedforward_dims is None:
            feedforward_dims = 4 * in_dims
        self.in_dims = in_dims
        self.feedforward_dims = feedforward_dims

        self.norm = nn.LayerNorm(in_dims)
        self.lin1 = nn.Linear(in_dims, feedforward_dims * 2)
        self.act1 = nn.GLU(dim=-1)

        self.kernel_size = kernel_size
        self.conv2 = nn.Conv2d(feedforward_dims, feedforward_dims, kernel_size=(kernel_size, kernel_size), bias=False,
                               stride=(1, 1), padding=(2, 2), padding_mode=_CONV_PADDING_MODE, groups=feedforward_dims)
        self.bn2 = nn.BatchNorm2d(feedforward_dims)
        self.act2 = act_layer()

        self.lin3 = nn.Linear(feedforward_dims, in_dims)
        self.drop = nn.Dropout(drop_prob)

        self.initialize_parameters()

    def initialize_parameters(self):
        kernel_size = self.conv2.kernel_size
        nn.init.normal_(self.conv2.weight, mean=0.0, std=math.sqrt(2 / (kernel_size[0] * kernel_size[1])))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.norm(x)
        x = self.lin1(x)
        x = self.act1(x)

        x = x.permute(0, 3, 1, 2)  # (n, h, w, c) -> (n, c, h, w)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = x.permute(0, 2, 3, 1)  # (n, c, h, w) -> (n, h, w, c)

        x = self.lin3(x)
        x = self.drop(x)

        x = x + identity
        return x


class PreNormOrderedReductionSA(nn.Module):

    def __init__(self,
                 in_dims: int,
                 num_heads: int,
                 reduction_ratio: int = 2,
                 shift_size: int = 0,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0):
        super().__init__()
        self.in_dims = in_dims
        self.num_heads = num_heads
        if in_dims % num_heads != 0:
            raise ValueError(f"Input dim {in_dims} is not divisible by num_heads {num_heads}.")
        self.head_dim = in_dims // num_heads

        # self.de_proj = nn.Linear(in_dims, in_dims, bias=False)
        # self.de_norm = nn.LayerNorm(in_dims)

        self.norm = nn.LayerNorm(in_dims)
        self.q_proj = nn.Linear(in_dims, in_dims)
        self.k_proj = nn.Linear(in_dims, in_dims)
        self.v_proj = nn.Linear(in_dims, in_dims)
        self.o_proj = nn.Linear(in_dims, in_dims)

        self.mean_proj = nn.Linear(in_dims, in_dims, bias=False)
        self.mean_norm = nn.LayerNorm(in_dims)

        self.attn_scale = math.sqrt(1 / self.head_dim)
        self.drop = nn.Dropout(drop_prob)
        self.attn_drop = nn.Dropout(attn_drop_prob)

        if reduction_ratio % 2 != 0:
            raise ValueError(f"Reduction ratio {reduction_ratio} should be even.")
        self.reduction_ratio = reduction_ratio
        if (shift_size > 0) and (shift_size != reduction_ratio // 2):
            raise ValueError(f"Shift size {shift_size} should be half of reduction_ratio {reduction_ratio}.")
        self.shift_size = shift_size

    def _reshape_4d(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, d = x.shape
        x = x.view(b, h * w, self.num_heads, d // self.num_heads).transpose(1, 2)
        return x.contiguous()

    def forward(self, x: torch.Tensor, de: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, w, d = x.shape
        assert d == self.in_dims

        identity = x

        x_norm = self.norm(x)
        # de = self.de_proj(de)
        # de = self.de_norm(de)

        # q = self.q_proj(x + de)  # (b, h, w, d)
        q = self.q_proj(x_norm)  # (b, h, w, d)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # de = torch.roll(de, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        if (h % self.reduction_ratio != 0) or (w % self.reduction_ratio != 0):
            raise ValueError(f"Input shape ({h}, {w}) not divisible by reduction ratio {self.reduction_ratio}.")

        r = self.reduction_ratio
        x_red = torch.mean(x.view(b, h // r, r, w // r, r, d), dim=[2, 4])
        # de_red = torch.mean(de.view(b, h // r, r, w // r, r, d), dim=[2, 4])

        # x_red = self.mean_act(self.mean_proj(x_red))
        x_red = self.mean_proj(x_red)
        x_red_norm = self.mean_norm(x_red)

        # k = self.k_proj(x_red + de_red)  # (b, h/r, w/r, d)
        k = self.k_proj(x_red_norm)  # (b, h/r, w/r, d)
        v = self.v_proj(x_red_norm)  # (b, h/r, w/r, d)

        q_flat = self._reshape_4d(q)  # (b, nh, hw, hd)
        k_flat = self._reshape_4d(k)  # (b, nh, hw/rr, hd)
        v_flat = self._reshape_4d(v)  # (b, nh, hw/rr, hd)

        attn = torch.matmul(q_flat, k_flat.transpose(-1, -2))  # (b, nh, hw, hw/rr)
        attn *= self.attn_scale
        attn = torch.softmax(attn, dim=-1)
        attn_drop = self.attn_drop(attn)

        out = torch.matmul(attn_drop, v_flat)  # (b, nh, hw, hd)
        out = out.transpose(1, 2).reshape(b, h, w, d)
        out = self.o_proj(out)
        out = self.drop(out)

        out = out + identity
        return out, attn


class OrderedReductionBlock(nn.Module):

    def __init__(self,
                 in_dims: int,
                 num_heads: int,
                 reduction_ratio: int = 8,
                 feedforward_dims: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0,
                 act_layer=nn.GELU):
        super().__init__()
        if reduction_ratio % 2 != 0:
            raise ValueError(f"Reduction error should be even, got {reduction_ratio}.")

        self.de_ff = nn.Sequential(
            nn.Linear(in_dims, in_dims * 4),
            nn.Dropout(drop_prob),
            act_layer(),
            nn.Linear(in_dims * 4, in_dims, bias=False),
        )
        # self.de_ff = nn.Linear(in_dims, in_dims, bias=False)
        self.de_norm = nn.LayerNorm(in_dims)
        nn.init.constant_(self.de_norm.weight, val=0.1)  # start with less impact

        sa_kwargs = dict(reduction_ratio=reduction_ratio, attn_drop_prob=attn_drop_prob, drop_prob=drop_prob)
        ff_kwargs = dict(feedforward_dims=feedforward_dims, drop_prob=drop_prob, act_layer=act_layer)

        self.sa1 = PreNormOrderedReductionSA(in_dims, num_heads, shift_size=0, **sa_kwargs)
        # self.ff1 = PreNormFF(in_dims, **ff_kwargs)
        self.ff1 = PreNormDWConvFF(in_dims, **ff_kwargs)
        # self.norm1 = nn.LayerNorm(in_dims, elementwise_affine=False)

        # self.sa2 = PreNormOrderedReductionSA(in_dims, num_heads, shift_size=reduction_ratio // 2, **sa_kwargs)
        self.sa2 = PreNormOrderedReductionSA(in_dims, num_heads, shift_size=0, **sa_kwargs)
        # self.ff2 = PreNormFF(in_dims, **ff_kwargs)
        self.ff2 = PreNormDWConvFF(in_dims, **ff_kwargs)
        self.norm2 = nn.LayerNorm(in_dims, elementwise_affine=True)

    def forward(self, x: torch.Tensor, de: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # b, h, w, c = x.shape

        de = self.de_ff(de)
        de = self.de_norm(de)
        x = x + de

        x, attn1 = self.sa1(x, de)
        x = self.ff1(x)
        # x = self.norm1(x)

        x, attn2 = self.sa2(x, de)
        x = self.ff2(x)
        x = self.norm2(x)

        return x, (attn1, attn2)


class OrderedReductionRegHead(nn.Module):

    def __init__(self,
                 in_dims: int,
                 num_heads: int,
                 num_repeats: int,
                 num_emb: int = 128,
                 reduction_ratio: int = 8,
                 feedforward_dims: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0,
                 act_layer=nn.GELU):
        super().__init__()
        self.in_dims = in_dims
        self.num_repeats = num_repeats

        conv_kwargs = dict(act_layer=act_layer, use_gn=False)
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                ConvBN(in_dims, in_dims // 4, 3, **conv_kwargs),
                ConvBN(in_dims // 4, in_dims // 4, 3, **conv_kwargs),
                nn.Conv2d(in_dims // 4, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
                # nn.Conv2d(in_dims // 4, 1, (3, 3), stride=(1, 1), padding=(1, 1), padding_mode=_CONV_PADDING_MODE),
            ) for _ in range(num_repeats + 1)
        ])

        self.attn_layers = nn.ModuleList([
            OrderedReductionBlock(in_dims, num_heads, reduction_ratio, act_layer=act_layer,
                                  feedforward_dims=feedforward_dims, attn_drop_prob=attn_drop_prob, drop_prob=drop_prob)
            for _ in range(num_repeats)
        ])

        self.log_sigmoid = nn.LogSigmoid()
        self.sigmoid = nn.Sigmoid()
        # self.drop = nn.Dropout(drop_prob)

        # initialize depth embedding with sinusoidal
        self.num_emb = num_emb
        with torch.no_grad():
            emb = torch.zeros(num_emb, in_dims, dtype=torch.float32)  # (n, d)
            pos = torch.arange(0, num_emb, dtype=torch.float32)
            # inv_freq = 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
            inv_freq = torch.exp(
                torch.arange(0.0, in_dims, 2.0, dtype=torch.float32).mul_(-math.log(2000.0) / in_dims))
            pos_dot = torch.outer(pos, inv_freq)  # (n,) x (d//2,) -> (n, d//2)
            emb[:, 0::2] = torch.sin(pos_dot)
            emb[:, 1::2] = torch.cos(pos_dot)
            emb *= math.sqrt(1 / in_dims)
        # self.depth_embedding = nn.Parameter(emb, requires_grad=True)
        self.register_buffer("depth_embedding", emb)

    @torch.no_grad()
    def _logit_to_indices(self, out: torch.Tensor) -> torch.Tensor:
        # b, c, h, w = out.shape
        assert out.shape[1] == 1  # 1-channel
        indices = self.log_sigmoid(out.detach())  # approx (-10 ~ 0)
        indices = (indices / 10).add(1).clamp(0, 1)  # (-10 ~ 0) -> (-1 ~ 0) -> (0 ~ 1) -> [0, 1)
        indices = torch.floor(indices * self.num_emb - 1e-3)  # [0, 128)
        indices = indices.long().squeeze(1)  # (b, h, w)
        return indices

    def forward(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        # b, h, w, c = x.shape

        outs = list()
        attn_weights = tuple()

        for i in range(self.num_repeats):
            x_chw = x.permute(0, 3, 1, 2)
            logit = self.conv_layers[i](x_chw)  # (b, 1, h, w)
            out = self.sigmoid(logit)
            outs.append(out)

            indices = self._logit_to_indices(logit)
            de = F.embedding(indices, self.depth_embedding)  # (b, h, w, d)
            # de = self.drop(de)

            x, aws = self.attn_layers[i](x, de)
            attn_weights += aws

        x_chw = x.permute(0, 3, 1, 2)
        logit = self.conv_layers[self.num_repeats](x_chw)  # (b, 1, h, w)
        out = self.sigmoid(logit)
        outs.append(out)

        outs = tuple(outs)
        return outs, attn_weights


class OrderedReductionRegDecoder(nn.Module):

    def __init__(self,
                 dec_dim: int = 512,
                 enc_dims: Tuple[int, int, int, int] = (192, 384, 768, 1536),
                 num_heads: int = 8,
                 num_repeats: int = 3,
                 num_emb: int = 128,
                 reduction_ratio: int = 8,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0,
                 act_layer=nn.GELU):
        super().__init__()

        self.dec_dim = dec_dim
        self.enc_dims = enc_dims
        assert len(enc_dims) == 4
        if dec_dim % 4 != 0:
            raise ValueError(f"Decoder dim {dec_dim} should be a multiple of 4.")

        # -------------------------------------------------------------- #
        # Neck
        # -------------------------------------------------------------- #

        conv_kwargs = dict(act_layer=act_layer, use_gn=False)
        self.enc_conv32 = nn.Sequential(
            ConvBN(enc_dims[3], enc_dims[3], 3, **conv_kwargs),
            ConvBN(enc_dims[3], dec_dim // 4, 3, **conv_kwargs),
            nn.UpsamplingBilinear2d(scale_factor=8)
        )
        self.enc_conv16 = nn.Sequential(
            ConvBN(enc_dims[2], enc_dims[2], 3, **conv_kwargs),
            ConvBN(enc_dims[2], dec_dim // 2, 3, **conv_kwargs),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.enc_conv8 = nn.Sequential(
            ConvBN(enc_dims[1], enc_dims[1], 3, **conv_kwargs),
            ConvBN(enc_dims[1], dec_dim, 3, **conv_kwargs),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.enc_conv4 = nn.Sequential(
            ConvBN(enc_dims[0], enc_dims[0], 3, **conv_kwargs),
            ConvBN(enc_dims[0], dec_dim * 2, 3, **conv_kwargs),
            nn.Identity()  # for consistency
        )

        enc_channels = (dec_dim // 4) + (dec_dim // 2) + dec_dim + (dec_dim * 2)
        # self.dec_linear = nn.Linear(enc_channels, dec_dim, bias=True)
        self.dec_linear = nn.Linear(enc_channels, dec_dim, bias=False)
        self.dec_norm = nn.LayerNorm(dec_dim, elementwise_affine=True)

        # -------------------------------------------------------------- #
        # Head
        # -------------------------------------------------------------- #

        self.reducer = OrderedReductionRegHead(
            dec_dim, num_heads, num_repeats, num_emb=num_emb, reduction_ratio=reduction_ratio,
            attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        for module in self.modules():
            # zero filling biases
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d) and (module.bias is not None):
                nn.init.zeros_(module.bias)

    def forward(self, enc_features: torch.Tensor
                ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Forward function."""
        e4, e8, e16, e32 = enc_features

        e32 = self.enc_conv32(e32)
        e16 = self.enc_conv16(e16)
        e8 = self.enc_conv8(e8)
        e4 = self.enc_conv4(e4)

        # 1/4 scale
        dec = torch.cat([e4, e8, e16, e32], dim=1)  # (b, c, h, w)
        # dec = self.dec_conv(dec)
        dec = dec.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)
        dec = self.dec_linear(dec)
        dec = self.dec_norm(dec)

        outs, attn_weights = self.reducer(dec)

        return outs, attn_weights
