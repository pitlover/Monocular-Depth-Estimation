from typing import Tuple, List
import math
import torch
import torch.nn as nn
# import torch.nn.functional as F

from .layer_utils import ConvBN, PyramidPoolingModule
from ..Depthformer.luna_layer import PreNormLunaLayer


class ODALunaDecoderRP(nn.Module):
    """Luna decoder stack, regression."""

    def __init__(self,
                 channels: int,  # 1024
                 input_channels: Tuple[int, int, int, int],
                 input_size: Tuple[int, int],  # same as Encoder
                 num_aux: int,  # 256
                 aux_dim: int,  # 256
                 num_heads: int,  # 8
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1,
                 output_channel: int = 1,
                 use_gn: bool = False,
                 num_groups: int = 1,
                 act_layer=nn.GELU):
        super().__init__()

        self.input_size = input_size
        self.channels = channels
        self.input_channels = input_channels
        assert len(input_channels) == 4
        self.output_channels = [max(channels // 8, aux_dim), channels // 4, channels // 2, channels]
        # [128, 256, 512, 1024]

        self.aux_dim = aux_dim
        self.num_heads = [max(num_aux // 8, 1), num_heads // 4, num_heads // 2, num_heads]
        self.aux = nn.Parameter(torch.zeros(1, num_aux, aux_dim))
        nn.init.normal_(self.aux, mean=0, std=math.sqrt(1 / aux_dim))

        pre_common_kwargs = dict(use_gn=use_gn, num_groups=num_groups, act_layer=act_layer)
        post_common_kwargs = dict(use_gn=use_gn, num_groups=num_groups, act_layer=None)

        self.ppm = PyramidPoolingModule(input_channels[3], input_channels[3], spatial_sizes=(1, 2, 3, 6))

        self.block32_pre = ConvBN(input_channels[3], self.output_channels[3], 3, **pre_common_kwargs)
        self.block32_luna = PreNormLunaLayer(
            self.output_channels[3], aux_dim=aux_dim, qk_proj_dim=min(self.output_channels[3], aux_dim),
            num_heads=self.num_heads[3], attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer
        )
        self.block32_post = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),  # 1/32 -> 1/16
            ConvBN(self.output_channels[3] // 4, self.output_channels[2], 1, **post_common_kwargs),
        )

        self.block16_pre = ConvBN(input_channels[2] + self.output_channels[2],
                                  self.output_channels[2], 3, **pre_common_kwargs)
        self.block16_luna = PreNormLunaLayer(
            self.output_channels[2], aux_dim=aux_dim, qk_proj_dim=min(self.output_channels[2], aux_dim),
            num_heads=self.num_heads[2], attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer
        )
        self.block16_post = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),  # 1/16 -> 1/8
            ConvBN(self.output_channels[2] // 4, self.output_channels[1], 1, **post_common_kwargs),
        )

        self.block8_pre = ConvBN(input_channels[1] + self.output_channels[1],
                                 self.output_channels[1], 3, **pre_common_kwargs)
        self.block8_luna = PreNormLunaLayer(
            self.output_channels[1], aux_dim=aux_dim, qk_proj_dim=min(self.output_channels[1], aux_dim),
            num_heads=self.num_heads[1], attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer
        )
        self.block8_post = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),  # 1/8 -> 1/4
            ConvBN(self.output_channels[1] // 4, self.output_channels[0], 1, **post_common_kwargs),
        )

        self.block4_pre = ConvBN(input_channels[0] + self.output_channels[0],
                                 self.output_channels[0], 3, **pre_common_kwargs)
        self.block4_luna = PreNormLunaLayer(
            self.output_channels[0], aux_dim=aux_dim, qk_proj_dim=min(self.output_channels[0], aux_dim),
            num_heads=self.num_heads[0], attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer
        )
        self.block4_post = nn.PixelShuffle(upscale_factor=2)

        final_channels = self.output_channels[0]  # [128,]
        self.block2 = nn.Sequential(
            ConvBN(self.output_channels[0] // 4, final_channels, 3, **pre_common_kwargs),
            nn.Conv2d(final_channels, output_channel, kernel_size=(1, 1))
        )

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        f0, f1, f2, f3 = features
        # f0: (b, 96x288, 192)
        # f1: (b, 48x144, 384)
        # f2: (b, 24x72, 768)
        # f3: (b,12x36, 1536)

        f0 = f0.view(-1, self.input_size[0] // 4, self.input_size[1] // 4, self.input_channels[0])
        f1 = f1.view(-1, self.input_size[0] // 8, self.input_size[1] // 8, self.input_channels[1])
        f2 = f2.view(-1, self.input_size[0] // 16, self.input_size[1] // 16, self.input_channels[2])
        f3 = f3.view(-1, self.input_size[0] // 32, self.input_size[1] // 32, self.input_channels[3])

        f0 = f0.permute(0, 3, 1, 2).contiguous()  # (b, 192, 96, 288)
        f1 = f1.permute(0, 3, 1, 2).contiguous()  # (b, 384, 48, 144)
        f2 = f2.permute(0, 3, 1, 2).contiguous()  # (b, 768, 24, 72)
        f3 = f3.permute(0, 3, 1, 2).contiguous()  # (b, 1536, 12, 36)

        aux = self.aux

        # -------- 1/32 scale -------- #
        f3 = self.ppm(f3)  # (b, 1526, 12, 36)
        c3 = self.block32_pre(f3)  # (b, 1536, 12, 36) -> (b, 1024, 12, 36)
        c3, aux, attn32_1, attn32_2 = self.block32_luna(c3, aux)
        c3 = self.block32_post(c3)  # (b, 256, 24, 72)

        # -------- 1/16 scale -------- #
        c2 = torch.cat([c3, f2], dim=1)  # (b, 256 + 768, 24, 72)
        c2 = self.block16_pre(c2)  # (b, 256 + 768, 24, 72) -> (b, 256, 24, 72)
        c2, aux, attn16_1, attn16_2 = self.block16_luna(c2, aux)
        c2 = self.block16_post(c2)  # (b, 64, 48, 144)

        # -------- 1/8 scale -------- #
        c1 = torch.cat([c2, f1], dim=1)  # (b, 64 + 384, 48, 144)
        c1 = self.block8_pre(c1)  # (b, 64 + 384, 48, 144) -> (b, 64, 48, 144)
        c1, aux, attn8_1, attn8_2 = self.block8_luna(c1, aux)
        c1 = self.block8_post(c1)  # -> (b, 16, 96, 288)

        # -------- 1/4 scale -------- #
        c0 = torch.cat([c1, f0], dim=1)  # (b, 16 + 192, 96, 288)
        c0 = self.block4_pre(c0)  # (b, 16 + 192, 96, 288) -> (b, 16, 96, 288)
        c0, aux, attn4_1, attn4_2 = self.block4_luna(c0, aux)
        c0 = self.block4_post(c0)  # (b, 4, 192, 576)

        # -------- 1/2 scale -------- #
        out = self.block2(c0)

        return out, aux, (attn4_1, attn4_2, attn8_1, attn8_2, attn16_1, attn16_2, attn32_1, attn32_2)
