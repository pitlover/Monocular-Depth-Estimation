from typing import Tuple, Optional
import math
import torch
import torch.nn as nn

from .oda2_layer_utils import ConvBN


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


class PreNormReductionSA(nn.Module):

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

        self.norm = nn.LayerNorm(in_dims)
        self.q_proj = nn.Linear(in_dims, in_dims)
        self.k_proj = nn.Linear(in_dims, in_dims)
        self.v_proj = nn.Linear(in_dims, in_dims)
        self.o_proj = nn.Linear(in_dims, in_dims)

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, w, d = x.shape
        assert d == self.in_dims

        identity = x

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x = self.norm(x)
        q = self.q_proj(x)  # (b, h, w, d)

        if (h % self.reduction_ratio != 0) or (w % self.reduction_ratio != 0):
            raise ValueError(f"Input shape ({h}, {w}) not divisible by reduction ratio {self.reduction_ratio}.")

        r = self.reduction_ratio
        x_half = torch.mean(x.view(b, h // r, r, w // r, r, d), dim=[2, 4])
        k = self.k_proj(x_half)  # (b, h/r, w/r, d)
        v = self.v_proj(x_half)  # (b, h/r, w/r, d)

        q_flat = self._reshape_4d(q)  # (b, nh, hw, hd)
        k_flat = self._reshape_4d(k)  # (b, nh, hw/rr, hd)
        v_flat = self._reshape_4d(v)  # (b, nh, hw/rr, hd)

        attn = torch.matmul(q_flat, k_flat.transpose(-1, -2))  # (b, nh, hw, hw/rr)
        attn *= self.attn_scale
        attn = torch.softmax(attn, dim=-1)
        # TODO maybe we need masking to handle edges
        attn_drop = self.attn_drop(attn)

        out = torch.matmul(attn_drop, v_flat)  # (b, nh, hw, hd)
        out = out.transpose(1, 2).reshape(b, h, w, d)
        out = self.o_proj(out)
        out = self.drop(out)

        if self.shift_size > 0:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        out = out + identity
        return out, attn


class IncrementalReductionModule(nn.Module):

    def __init__(self,
                 in_dims: int,
                 num_heads: int,
                 feedforward_dims: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0,
                 act_layer=nn.GELU):
        super().__init__()

        self.sa8_1 = PreNormReductionSA(in_dims, num_heads, reduction_ratio=8, shift_size=0,
                                        attn_drop_prob=attn_drop_prob, drop_prob=drop_prob)
        self.ff8_1 = PreNormFF(in_dims, drop_prob=drop_prob, feedforward_dims=feedforward_dims, act_layer=act_layer)

        self.sa8_2 = PreNormReductionSA(in_dims, num_heads, reduction_ratio=8, shift_size=4,
                                        attn_drop_prob=attn_drop_prob, drop_prob=drop_prob)
        self.ff8_2 = PreNormFF(in_dims, drop_prob=drop_prob, feedforward_dims=feedforward_dims, act_layer=act_layer)

        self.sa4_1 = PreNormReductionSA(in_dims, num_heads, reduction_ratio=4, shift_size=0,
                                        attn_drop_prob=attn_drop_prob, drop_prob=drop_prob)
        self.ff4_1 = PreNormFF(in_dims, drop_prob=drop_prob, feedforward_dims=feedforward_dims, act_layer=act_layer)

        self.sa4_2 = PreNormReductionSA(in_dims, num_heads, reduction_ratio=4, shift_size=2,
                                        attn_drop_prob=attn_drop_prob, drop_prob=drop_prob)
        self.ff4_2 = PreNormFF(in_dims, drop_prob=drop_prob, feedforward_dims=feedforward_dims, act_layer=act_layer)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        x, attn8_1 = self.sa8_1(x)
        x = self.ff8_1(x)
        x, attn8_2 = self.sa8_2(x)
        x = self.ff8_2(x)

        x, attn4_1 = self.sa4_1(x)
        x = self.ff4_1(x)
        x, attn4_2 = self.sa4_2(x)
        x = self.ff4_2(x)

        return x, (attn8_1, attn8_2, attn4_1, attn4_2)


class ReductionTransformerRegDecoder(nn.Module):

    def __init__(self,
                 dec_dim: int = 512,
                 enc_dims: Tuple[int, int, int, int] = (192, 384, 768, 1536),
                 num_heads: int = 16,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0,
                 act_layer=nn.GELU):
        super().__init__()

        self.dec_dim = dec_dim
        self.enc_dims = enc_dims
        assert len(enc_dims) == 4
        if dec_dim % 4 != 0:
            raise ValueError(f"Decoder dim {dec_dim} should be a multiple of 4.")

        self.enc_conv32 = nn.Sequential(
            ConvBN(enc_dims[3], enc_dims[3], 3, act_layer=act_layer),
            ConvBN(enc_dims[3], dec_dim // 4, 3, act_layer=act_layer),
            nn.UpsamplingBilinear2d(scale_factor=8)
        )
        self.enc_conv16 = nn.Sequential(
            ConvBN(enc_dims[2], enc_dims[2], 3, act_layer=act_layer),
            ConvBN(enc_dims[2], dec_dim // 2, 3, act_layer=act_layer),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.enc_conv8 = nn.Sequential(
            ConvBN(enc_dims[1], enc_dims[1], 3, act_layer=act_layer),
            ConvBN(enc_dims[1], dec_dim, 3, act_layer=act_layer),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.enc_conv4 = nn.Sequential(
            ConvBN(enc_dims[0], enc_dims[0], 3, act_layer=act_layer),
            ConvBN(enc_dims[0], dec_dim * 2, 3, act_layer=act_layer),
            nn.Identity()  # for consistency
        )

        enc_channels = (dec_dim // 4) + (dec_dim // 2) + dec_dim + (dec_dim * 2)
        # self.dec_conv = ConvBN(enc_channels, dec_dim, 3, act_layer=act_layer)
        self.dec_linear = nn.Linear(enc_channels, dec_dim, bias=False)

        self.norm = nn.LayerNorm(dec_dim)
        self.reducer = IncrementalReductionModule(
            dec_dim, num_heads, attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer)

        self.out_conv = nn.Sequential(
            ConvBN(dec_dim, dec_dim // 4, 3, act_layer=act_layer),
            nn.Conv2d(dec_dim // 4, 1, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
        )
        self.out_act = nn.Sigmoid()

    def forward(self, enc_features: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
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
        dec = self.norm(dec)

        dec, attn_weights = self.reducer(dec)
        dec = dec.permute(0, 3, 1, 2).contiguous()

        out = self.out_conv(dec)
        out = self.out_act(out)

        return out, attn_weights
