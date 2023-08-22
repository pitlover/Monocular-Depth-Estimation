from typing import Tuple, List
import torch
import torch.nn as nn

from .oda2_layer_utils import PyramidPoolingModule, ConvBN


class ODA2ConvDecoder(nn.Module):
    """Basic Conv decoder stack, regression."""

    def __init__(self,
                 channels: int,
                 input_channels: Tuple[int, int, int, int],
                 output_channel: int = 1,
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
            input_channels[-1], channels // 2, channels, spatial_sizes=(1, 2, 3, 6),
            act_layer=act_layer, **act_kwargs
        )

        self.block32 = nn.Sequential(
            ConvBN(channels, self.output_channels[3], 3, act_layer=act_layer, **act_kwargs),
            ConvBN(self.output_channels[3], self.output_channels[3], 3, act_layer=act_layer, **act_kwargs),
            nn.UpsamplingBilinear2d(scale_factor=2),  # 1/32 -> 1/16
            ConvBN(self.output_channels[3], self.output_channels[2], 1, act_layer=None),
        )

        self.block16 = nn.Sequential(
            ConvBN(input_channels[2] + self.output_channels[2],
                   self.output_channels[2], 3, act_layer=act_layer),
            ConvBN(self.output_channels[2], self.output_channels[2], 3, act_layer=act_layer, **act_kwargs),
            nn.UpsamplingBilinear2d(scale_factor=2),  # 1/16 -> 1/8
            ConvBN(self.output_channels[2], self.output_channels[1], 1, act_layer=None),
        )

        self.block8 = nn.Sequential(
            ConvBN(input_channels[1] + self.output_channels[1],
                   self.output_channels[1], 3, act_layer=act_layer),
            ConvBN(self.output_channels[1], self.output_channels[1], 3, act_layer=act_layer, **act_kwargs),
            nn.UpsamplingBilinear2d(scale_factor=2),  # 1/8 -> 1/4
            ConvBN(self.output_channels[1], self.output_channels[0], 1, act_layer=None),
        )

        self.block4 = nn.Sequential(
            ConvBN(input_channels[0] + self.output_channels[0],
                   self.output_channels[0], 3, act_layer=act_layer),
            ConvBN(self.output_channels[0], self.output_channels[0], 3, act_layer=act_layer, **act_kwargs),
            nn.UpsamplingBilinear2d(scale_factor=2)  # 1/4 -> 1/2
        )

        final_channels = self.output_channels[0]  # [128,]
        self.block2 = nn.Sequential(
            ConvBN(self.output_channels[0], final_channels, 3, act_layer=act_layer, **act_kwargs),
            nn.Conv2d(final_channels, output_channel, kernel_size=(1, 1))
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        f0, f1, f2, f3 = features
        # f0: (b, 192, 96, 288)
        # f1: (b, 384, 48, 144)
        # f2: (b, 768, 24, 72)
        # f3: (b, 1536, 12, 36)

        # -------- 1/32 scale -------- #
        c3 = self.ppm(f3)
        c3 = self.block32(c3)

        # -------- 1/16 scale -------- #
        c2 = torch.cat([c3, f2], dim=1)
        c2 = self.block16(c2)

        # -------- 1/8 scale -------- #
        c1 = torch.cat([c2, f1], dim=1)
        c1 = self.block8(c1)

        # -------- 1/4 scale -------- #
        c0 = torch.cat([c1, f0], dim=1)
        c0 = self.block4(c0)

        # -------- 1/2 scale -------- #
        out = self.block2(c0)

        return out
