from typing import Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .oda2_layer_utils import PyramidPoolingModule, ConvBN


class ODA2LunaLayer(nn.Module):

    def __init__(self,
                 in_dims: int,
                 out_dims: int,
                 aux_dims: int,
                 num_heads: int,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1,
                 act_layer=nn.GELU):
        super().__init__()

        self.in_dims = in_dims
        self.aux_dims = aux_dims
        self.num_heads = num_heads
        self.head_dim = aux_dims // num_heads

        # self attention
        self.q_self = nn.Linear(aux_dims, aux_dims)
        self.k_self = nn.Linear(aux_dims, aux_dims)
        self.v_self = nn.Linear(aux_dims, aux_dims)
        self.o_self = nn.Linear(aux_dims, aux_dims)
        self.norm_self = nn.LayerNorm(aux_dims)

        # cross attention 1
        self.q_cross1 = nn.Linear(aux_dims, aux_dims)
        self.k_cross1 = nn.Linear(in_dims, aux_dims)
        self.v_cross1 = nn.Linear(in_dims, aux_dims)
        self.o_cross1 = nn.Linear(aux_dims, aux_dims)
        self.norm_cross1 = nn.LayerNorm(aux_dims)

        # cross attention 2
        self.q_cross2 = nn.Linear(in_dims, aux_dims)
        self.k_cross2 = nn.Linear(aux_dims, aux_dims)
        self.v_cross2 = nn.Linear(aux_dims, out_dims)
        self.o_cross2 = nn.Linear(out_dims, out_dims)

        self.drop = nn.Dropout(drop_prob)
        self.attn_drop = nn.Dropout(attn_drop_prob)

        # FF
        self.ff = nn.Sequential(
            nn.Linear(aux_dims, aux_dims * 4),
            act_layer(),
            nn.Dropout(drop_prob),
            nn.Linear(aux_dims * 4, aux_dims),
        )
        self.norm_ff = nn.LayerNorm(aux_dims)

        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.zeros_(self.o_cross2.weight)  # zero at first
        # nn.init.uniform_(self.o_cross2.weight, -0.01, 0.01)  # zero at first
        for param_name, param in self.named_parameters():
            if "bias" in param_name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        _, s, d = aux.shape
        assert (c == self.in_dims) and (d == self.aux_dims)
        head_dim = d // self.num_heads
        hw = h * w

        # self attention
        q_self = self.q_self(aux)
        k_self = self.k_self(aux)
        v_self = self.v_self(aux)

        q_self = q_self.view(b, s, self.num_heads, head_dim).transpose(1, 2)
        k_self = k_self.view(b, s, self.num_heads, head_dim).transpose(1, 2)
        v_self = v_self.view(b, s, self.num_heads, head_dim).transpose(1, 2)
        attn_self = torch.matmul(q_self, k_self.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_self = torch.softmax(attn_self, dim=-1)
        attn_self_drop = self.attn_drop(attn_self)
        aux_c = torch.matmul(attn_self_drop, v_self).transpose(1, 2).reshape(b, s, d)
        aux_c = self.o_self(aux_c)
        aux_c = self.drop(aux_c)

        aux = self.norm_self(aux + aux_c)

        # cross attention 1
        x = x.permute(0, 2, 3, 1).reshape(b, -1, c)

        q_cross1 = self.q_cross1(aux)
        k_cross1 = self.k_cross1(x)
        v_cross1 = self.v_cross1(x)

        q_cross1 = q_cross1.view(b, s, self.num_heads, head_dim).transpose(1, 2)
        k_cross1 = k_cross1.view(b, hw, self.num_heads, head_dim).transpose(1, 2)
        v_cross1 = v_cross1.view(b, hw, self.num_heads, head_dim).transpose(1, 2)
        attn_cross1 = torch.matmul(q_cross1, k_cross1.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_cross1 = torch.softmax(attn_cross1, dim=-1)
        attn_cross1_drop = self.attn_drop(attn_cross1)
        aux_c = torch.matmul(attn_cross1_drop, v_cross1).transpose(1, 2).reshape(b, s, d)
        aux_c = self.o_cross1(aux_c)
        aux_c = self.drop(aux_c)

        aux = self.norm_cross1(aux + aux_c)

        # ff
        aux_ff = self.ff(aux)
        aux_ff = self.drop(aux_ff)
        aux = self.norm_ff(aux + aux_ff)

        # cross attention 2
        q_cross2 = self.q_cross2(x)
        k_cross2 = self.k_cross2(aux)
        v_cross2 = self.v_cross2(aux)

        q_cross2 = q_cross2.view(b, hw, self.num_heads, head_dim).transpose(1, 2)
        k_cross2 = k_cross2.view(b, s, self.num_heads, head_dim).transpose(1, 2)
        v_cross2 = v_cross2.view(b, s, self.num_heads, -1).transpose(1, 2)
        attn_cross2 = torch.matmul(q_cross2, k_cross2.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_cross2 = torch.softmax(attn_cross2, dim=-1)
        attn_cross2_drop = self.attn_drop(attn_cross2)
        weight_c = torch.matmul(attn_cross2_drop, v_cross2).transpose(1, 2).reshape(b, hw, -1)
        weight_c = self.o_cross2(weight_c)

        weight_c = weight_c.transpose(1, 2).reshape(b, -1, h, w)

        return aux, weight_c


class ODA2LunaGating(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 aux_dims: int,
                 num_heads: int,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1,
                 act_layer=nn.GELU):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=True)
        self.luna = ODA2LunaLayer(in_channels, out_channels, aux_dims, num_heads,
                                  attn_drop_prob, drop_prob, act_layer=act_layer)

        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), bias=False)
        # self.norm_out = nn.GroupNorm(32, out_channels)
        self.norm_out = nn.BatchNorm2d(out_channels)
        self.act = act_layer()

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_conv = self.conv(x)
        aux, x_weight = self.luna(x, aux)
        y = x_conv * torch.sigmoid(x_weight)

        y = self.conv_out(y)
        y = self.norm_out(y)
        y = self.act(y)
        return y, aux


class ODA2LunaRegDecoder(nn.Module):
    """Basic Luna decoder stack, regression."""

    def __init__(self,
                 channels: int,
                 input_channels: Tuple[int, int, int, int],
                 num_aux: int,
                 aux_dims: int,
                 num_heads: int,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1,
                 act_layer=nn.GELU,
                 **act_kwargs):
        super().__init__()

        if act_layer == nn.LeakyReLU:
            act_kwargs["negative_slope"] = 0.2

        self.channels = channels
        self.input_channels = input_channels
        assert len(input_channels) == 4
        self.output_channels = [channels // 8, channels // 4, channels // 2, channels]
        # [128, 256, 512, 1024]

        self.ppm = PyramidPoolingModule(
            input_channels[-1], 512, channels, spatial_sizes=(1, 2, 3, 6),
            act_layer=act_layer, **act_kwargs
        )

        self.num_aux = num_aux
        self.aux_dims = aux_dims
        self.aux = nn.Parameter(torch.zeros(1, num_aux, aux_dims))
        nn.init.trunc_normal_(self.aux, std=math.sqrt(1 / aux_dims))

        luna_kwargs = dict(aux_dims=aux_dims, num_heads=num_heads,
                           attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer)

        # convbn_kwargs = dict(act_layer=act_layer, use_gn=True, num_groups=8)
        convbn_kwargs = dict(act_layer=act_layer, use_gn=False)
        convbn_kwargs.update(act_kwargs)

        self.block32 = nn.Sequential(
            ConvBN(self.output_channels[3], self.output_channels[3], 3, **convbn_kwargs),
            ConvBN(self.output_channels[3], self.output_channels[3], 3, **convbn_kwargs),
            # ConvBN(self.output_channels[3], self.output_channels[3], 1, act_layer=None),
            # nn.UpsamplingBilinear2d(scale_factor=2),  # 1/32 -> 1/16
        )
        self.block16_lateral = ConvBN(input_channels[2], self.output_channels[3], 3, **convbn_kwargs)
        self.block16_gate = ODA2LunaGating(
            self.output_channels[3] * 2, self.output_channels[2], **luna_kwargs
        )

        self.block16 = nn.Sequential(
            ConvBN(self.output_channels[2], self.output_channels[2], 3, **convbn_kwargs),
            ConvBN(self.output_channels[2], self.output_channels[2], 3, **convbn_kwargs),
            # ConvBN(self.output_channels[2], self.output_channels[2], 1, act_layer=None),
            # nn.UpsamplingBilinear2d(scale_factor=2),  # 1/16 -> 1/8
        )
        self.block8_lateral = ConvBN(input_channels[1], self.output_channels[2], 3, **convbn_kwargs)
        self.block8_gate = ODA2LunaGating(
            self.output_channels[2] * 2, self.output_channels[1], **luna_kwargs
        )

        self.block8 = nn.Sequential(
            ConvBN(self.output_channels[1], self.output_channels[1], 3, **convbn_kwargs),
            ConvBN(self.output_channels[1], self.output_channels[1], 3, **convbn_kwargs),
            # ConvBN(self.output_channels[1], self.output_channels[1], 1, act_layer=None),
            # nn.UpsamplingBilinear2d(scale_factor=2),  # 1/8 -> 1/4
        )
        self.block4_lateral = ConvBN(input_channels[0], self.output_channels[1], 3, **convbn_kwargs)
        self.block4_gate = ODA2LunaGating(
            self.output_channels[1] * 2, self.output_channels[0], **luna_kwargs
        )

        self.block4 = nn.Sequential(
            ConvBN(self.output_channels[0], self.output_channels[0], 3, **convbn_kwargs),
            # ConvBN(self.output_channels[0], self.output_channels[0], 3, act_layer=act_layer, **act_kwargs),
            nn.Conv2d(self.output_channels[0], 1, kernel_size=(3, 3), padding=(1, 1), padding_mode="replicate")
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        f0, f1, f2, f3 = features
        # f0: (b, 192, 96, 288)
        # f1: (b, 384, 48, 144)
        # f2: (b, 768, 24, 72)
        # f3: (b, 1536, 12, 36)

        b = f0.shape[0]
        aux = self.aux
        aux = aux * math.sqrt(1 / self.aux_dims)
        aux = aux.expand(b, self.num_aux, self.aux_dims)

        # -------- 1/32 scale -------- #
        f3 = self.ppm(f3)
        c3 = self.block32(f3)
        c3 = F.interpolate(c3, scale_factor=2, mode="bilinear", align_corners=True)

        # -------- 1/16 scale -------- #
        f2 = self.block16_lateral(f2)
        c2 = torch.cat([c3, f2], dim=1)
        c2, aux = self.block16_gate(c2, aux)
        c2 = self.block16(c2)
        c2 = F.interpolate(c2, scale_factor=2, mode="bilinear", align_corners=True)

        # -------- 1/8 scale -------- #
        f1 = self.block8_lateral(f1)
        c1 = torch.cat([c2, f1], dim=1)
        c1, aux = self.block8_gate(c1, aux)
        c1 = self.block8(c1)
        c1 = F.interpolate(c1, scale_factor=2, mode="bilinear", align_corners=True)

        # -------- 1/4 scale -------- #
        f0 = self.block4_lateral(f0)
        c0 = torch.cat([c1, f0], dim=1)
        c0, aux = self.block4_gate(c0, aux)
        out = self.block4(c0)
        out = torch.sigmoid(out)

        return out
