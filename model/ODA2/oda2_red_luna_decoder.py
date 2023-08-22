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


def _reshape_4d(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, h, w, d = x.shape
    x = x.view(b, h * w, num_heads, d // num_heads).transpose(1, 2)
    return x.contiguous()


def _reshape_3d(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, n, d = x.shape
    x = x.view(b, n, num_heads, d // num_heads).transpose(1, 2)
    return x.contiguous()


class PreNormLunaS1(nn.Module):

    def __init__(self,
                 in_dims: int,
                 num_heads: int,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0):
        super().__init__()
        self.in_dims = in_dims
        self.num_heads = num_heads
        if in_dims % num_heads != 0:
            raise ValueError(f"Input dim {in_dims} is not divisible by num_heads {num_heads}.")
        self.head_dim = in_dims // num_heads

        self.norm = nn.LayerNorm(in_dims)
        self.aux_norm = nn.LayerNorm(in_dims)

        self.q_proj = nn.Linear(in_dims, in_dims)
        self.k_proj = nn.Linear(in_dims, in_dims)
        self.v_proj = nn.Linear(in_dims, in_dims)
        self.o_proj = nn.Linear(in_dims, in_dims)

        self.attn_scale = math.sqrt(1 / self.head_dim)
        self.drop = nn.Dropout(drop_prob)
        self.attn_drop = nn.Dropout(attn_drop_prob)

    def _reshape_4d(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, d = x.shape
        x = x.view(b, h * w, self.num_heads, d // self.num_heads).transpose(1, 2)
        return x.contiguous()

    def _reshape_3d(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        x = x.view(b, n, self.num_heads, d // self.num_heads).transpose(1, 2)
        return x.contiguous()

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, w, d = x.shape
        _, n, aux_d = aux.shape
        assert d == aux_d == self.in_dims

        identity = aux

        x = self.norm(x)
        aux = self.aux_norm(aux)

        q = self.q_proj(aux)  # (b, n, d)
        k = self.k_proj(x)  # (b, h, w, d)
        v = self.v_proj(x)  # (b, h, w, d)

        q_flat = _reshape_3d(q, self.num_heads)  # (b, nh, n, hd)
        k_flat = _reshape_4d(k, self.num_heads)  # (b, nh, hw, hd)
        v_flat = _reshape_4d(v, self.num_heads)  # (b, nh, hw, hd)

        attn = torch.matmul(q_flat, k_flat.transpose(-1, -2))  # (b, nh, n, hw)
        attn *= self.attn_scale
        attn = torch.softmax(attn, dim=-1)
        attn_drop = self.attn_drop(attn)

        out = torch.matmul(attn_drop, v_flat)  # (b, nh, n, hd)
        out = out.transpose(1, 2).reshape(b, n, d)
        out = self.o_proj(out)
        out = self.drop(out)

        out = out + identity  # new aux
        return out, attn


class PreNormLunaS2(nn.Module):

    def __init__(self,
                 in_dims: int,
                 num_heads: int,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0):
        super().__init__()
        self.in_dims = in_dims
        self.num_heads = num_heads
        if in_dims % num_heads != 0:
            raise ValueError(f"Input dim {in_dims} is not divisible by num_heads {num_heads}.")
        self.head_dim = in_dims // num_heads

        self.norm = nn.LayerNorm(in_dims)
        self.aux_norm = nn.LayerNorm(in_dims)

        self.q_proj = nn.Linear(in_dims, in_dims)
        self.k_proj = nn.Linear(in_dims, in_dims)
        self.v_proj = nn.Linear(in_dims, in_dims)
        self.o_proj = nn.Linear(in_dims, in_dims)

        self.attn_scale = math.sqrt(1 / self.head_dim)
        self.drop = nn.Dropout(drop_prob)
        self.attn_drop = nn.Dropout(attn_drop_prob)

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, w, d = x.shape
        _, n, aux_d = aux.shape
        assert d == aux_d == self.in_dims

        identity = x

        x = self.norm(x)
        aux = self.aux_norm(aux)

        q = self.q_proj(x)  # (b, h, w, d)
        k = self.k_proj(aux)  # (b, n, d)
        v = self.v_proj(aux)  # (b, n, d)

        q_flat = _reshape_4d(q, self.num_heads)  # (b, nh, hw, hd)
        k_flat = _reshape_3d(k, self.num_heads)  # (b, nh, n, hd)
        v_flat = _reshape_3d(v, self.num_heads)  # (b, nh, n, hd)

        attn = torch.matmul(q_flat, k_flat.transpose(-1, -2))  # (b, nh, hw, n)
        attn *= self.attn_scale
        attn = torch.softmax(attn, dim=-1)
        attn_drop = self.attn_drop(attn)

        out = torch.matmul(attn_drop, v_flat)  # (b, nh, hw`, hd)
        out = out.transpose(1, 2).reshape(b, h, w, d)
        out = self.o_proj(out)
        out = self.drop(out)

        out = out + identity  # new x
        return out, attn


class LunaModule(nn.Module):

    def __init__(self,
                 in_dims: int,
                 num_heads: int,
                 feedforward_dims: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0,
                 act_layer=nn.GELU):
        super().__init__()

        luna_kwargs = dict(attn_drop_prob=attn_drop_prob, drop_prob=drop_prob)
        ff_kwargs = dict(drop_prob=drop_prob, act_layer=act_layer, feedforward_dims=feedforward_dims)

        self.luna1 = PreNormLunaS1(in_dims, num_heads, **luna_kwargs)
        self.ff_aux = PreNormFF(in_dims, **ff_kwargs)
        self.luna2 = PreNormLunaS2(in_dims, num_heads, **luna_kwargs)
        self.ff = PreNormFF(in_dims, **ff_kwargs)

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        aux, attn_aux_to_x = self.luna1(x, aux)
        aux = self.ff_aux(aux)

        x, attn_x_to_aux = self.luna2(x, aux)
        x = self.ff(x)
        return x, aux, attn_aux_to_x, attn_x_to_aux


class StackedLunaModule(nn.Module):

    def __init__(self,
                 in_dims: int,
                 num_heads: int,
                 num_layers: int,
                 feedforward_dims: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0,
                 act_layer=nn.GELU):
        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            LunaModule(in_dims, num_heads, feedforward_dims=feedforward_dims,
                       attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer)
            for _ in range(num_layers)
        ])

    def forward(self,
                x: torch.Tensor,
                aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        attn = []
        for layer in self.layers:
            layer: LunaModule
            x, aux, attn1, attn2 = layer(x, aux)
            attn.append(attn1)
            attn.append(attn2)

        return x, aux, tuple(attn)


class LunaTransformerRegDecoder(nn.Module):

    def __init__(self,
                 dec_dim: int = 512,
                 enc_dims: Tuple[int, int, int, int] = (192, 384, 768, 1536),
                 num_aux: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0,
                 act_layer=nn.GELU):
        super().__init__()

        self.dec_dim = dec_dim
        self.enc_dims = enc_dims
        assert len(enc_dims) == 4
        if dec_dim % 4 != 0:
            raise ValueError(f"Decoder dim {dec_dim} should be a multiple of 4.")
        self.num_aux = num_aux

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
        self.dec_norm = nn.LayerNorm(dec_dim)

        # Sinusoidal positional encoding initialization because we want aux to have order.
        with torch.no_grad():
            aux = torch.zeros(num_aux, dec_dim, dtype=torch.float32)  # (n, d)
            pos = torch.arange(0, num_aux, dtype=torch.float32)
            # inv_freq = 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
            inv_freq = torch.exp(
                torch.arange(0.0, dec_dim, 2.0, dtype=torch.float32).mul_(-math.log(10000.0) / dec_dim))
            pos_dot = torch.outer(pos, inv_freq)  # (n,) x (d//2,) -> (n, d//2)
            aux[:, 0::2] = torch.sin(pos_dot)
            aux[:, 1::2] = torch.cos(pos_dot)
            aux = aux.unsqueeze(0)  # (n, d) -> (1, n, d)
        # self.aux = nn.Parameter(aux)  # TODO try parameterization
        self.register_buffer("aux", aux)

        self.enc_to_aux = nn.Linear(enc_channels, dec_dim, bias=True)
        self.aux_linear1 = nn.Linear(dec_dim, dec_dim, bias=True)
        self.aux_act = nn.Sigmoid()
        self.aux_linear2 = nn.Linear(dec_dim, dec_dim, bias=False)
        self.aux_norm = nn.LayerNorm(dec_dim)

        self.luna = StackedLunaModule(
            dec_dim, num_heads, num_layers, attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer)

        self.out_conv = nn.Sequential(
            ConvBN(dec_dim, dec_dim // 4, 3, act_layer=act_layer),
            nn.Conv2d(dec_dim // 4, 1, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
        )
        self.out_act = nn.Sigmoid()

    def forward(self, enc_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward function."""
        e4, e8, e16, e32 = enc_features

        e32 = self.enc_conv32(e32)
        e16 = self.enc_conv16(e16)
        e8 = self.enc_conv8(e8)
        e4 = self.enc_conv4(e4)

        # decoder input
        enc = torch.cat([e4, e8, e16, e32], dim=1)  # (b, c, h, w)
        enc = enc.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)
        dec = self.dec_linear(enc)
        dec = self.dec_norm(dec)
        b, h, w, d = dec.shape

        # aux
        aux = self.aux.expand(b, self.num_aux, d)  # (1, n, d) -> (b, n, d)
        aux = self.aux_linear1(aux)
        enc_mean = torch.mean(enc, dim=[1, 2])  # (b, h, w, c) -> (b, c)
        aux_weight = self.enc_to_aux(enc_mean).unsqueeze(1)  # (b, c) -> (b, d) -> (b, 1, d)
        aux = aux * self.aux_act(aux_weight)  # (b, n, d)
        aux = self.aux_linear2(aux)
        aux = self.aux_norm(aux)

        # luna
        dec, aux, attn_weights = self.luna(dec, aux)
        dec = dec.permute(0, 3, 1, 2).contiguous()

        out = self.out_conv(dec)
        out = self.out_act(out)

        return out, aux, attn_weights
