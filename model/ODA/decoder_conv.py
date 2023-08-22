from typing import Tuple, List
import torch
import torch.nn as nn

from .layer_utils import ConvBN


class ODAConvDecoder(nn.Module):
    """Basic Conv decoder stack, regression."""

    def __init__(self,
                 channels: int,
                 input_channels: Tuple[int, int, int, int],
                 input_size: Tuple[int, int],  # same as Encoder
                 output_channel: int = 1,
                 act_layer=nn.GELU):
        super().__init__()

        self.input_size = input_size
        self.channels = channels
        self.input_channels = input_channels
        assert len(input_channels) == 4
        self.output_channels = [channels // 8, channels // 4, channels // 2, channels]
        # [128, 256, 512, 1024]

        self.block32 = nn.Sequential(
            ConvBN(input_channels[3], self.output_channels[3], 3, act_layer=act_layer),
            ConvBN(self.output_channels[3], self.output_channels[3], 3, act_layer=act_layer),
            nn.UpsamplingBilinear2d(scale_factor=2),  # 1/32 -> 1/16
            ConvBN(self.output_channels[3], self.output_channels[2], 1, act_layer=None),
        )

        self.block16 = nn.Sequential(
            ConvBN(input_channels[2] + self.output_channels[2],
                   self.output_channels[2], 3, act_layer=act_layer),
            ConvBN(self.output_channels[2], self.output_channels[2], 3, act_layer=act_layer),
            nn.UpsamplingBilinear2d(scale_factor=2),  # 1/16 -> 1/8
            ConvBN(self.output_channels[2], self.output_channels[1], 1, act_layer=None),
        )

        self.block8 = nn.Sequential(
            ConvBN(input_channels[1] + self.output_channels[1],
                   self.output_channels[1], 3, act_layer=act_layer),
            ConvBN(self.output_channels[1], self.output_channels[1], 3, act_layer=act_layer),
            nn.UpsamplingBilinear2d(scale_factor=2),  # 1/8 -> 1/4
            ConvBN(self.output_channels[1], self.output_channels[0], 1, act_layer=None),
        )

        self.block4 = nn.Sequential(
            ConvBN(input_channels[0] + self.output_channels[0],
                   self.output_channels[0], 3, act_layer=act_layer),
            ConvBN(self.output_channels[0], self.output_channels[0], 3, act_layer=act_layer),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        final_channels = self.output_channels[0]  # [128,]
        self.block2 = nn.Sequential(
            ConvBN(self.output_channels[0], final_channels, 3, act_layer=act_layer),
            nn.Conv2d(final_channels, output_channel, kernel_size=(1, 1))
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
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

        # -------- 1/32 scale -------- #
        c3 = self.block32(f3)  # (b, 1536, 12, 36) -> (b, 1024, 12, 36) -> (b, 256, 24, 72)

        # -------- 1/16 scale -------- #
        c2 = torch.cat([c3, f2], dim=1)  # (b, 256 + 768, 24, 72)
        c2 = self.block16(c2)  # (b, 256 + 768, 24, 72) -> (b, 256, 24, 72) -> (b, 64, 48, 144)

        # -------- 1/8 scale -------- #
        c1 = torch.cat([c2, f1], dim=1)  # (b, 64 + 384, 48, 144)
        c1 = self.block8(c1)  # (b, 64 + 384, 48, 144) -> (b, 64, 48, 144) -> (b, 16, 96, 288)

        # -------- 1/4 scale -------- #
        c0 = torch.cat([c1, f0], dim=1)  # (b, 16 + 192, 96, 288)
        c0 = self.block4(c0)  # (b, 16 + 192, 96, 288) -> (b, 16, 96, 288) -> (b, 4, 192, 576)

        # -------- 1/2 scale -------- #
        out = self.block2(c0)  # (b, 4, 192, 576) -> (b, 1, 192, 576)

        return out
