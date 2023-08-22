from typing import Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_utils import ConvBN, PyramidPoolingModuleV2
from .lion_layer import LionLayer


class ODALionDecoder(nn.Module):
    """Lion decoder stack, regression."""

    def __init__(self,
                 channels: int,  # 2048
                 input_channels: Tuple[int, int, int, int],
                 input_size: Tuple[int, int],  # same as Encoder
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1,
                 output_channel: int = 1,
                 act_layer=nn.GELU):
        super().__init__()

        self.input_size = input_size
        self.channels = channels
        self.input_channels = input_channels
        assert len(input_channels) == 4
        # [128, 256, 512, 1024]

        self.hidden_dims = [channels // 8, channels // 4, channels // 2, channels]

        h_div2, w_div2 = input_size[0] // 2, input_size[1] // 2
        h_div4, w_div4 = input_size[0] // 4, input_size[1] // 4
        h_div8, w_div8 = input_size[0] // 8, input_size[1] // 8
        h_div16, w_div16 = input_size[0] // 16, input_size[1] // 16
        h_div32, w_div32 = input_size[0] // 32, input_size[1] // 32

        self.size2 = (h_div2, w_div2)
        self.size4 = (h_div4, w_div4)
        self.size8 = (h_div8, w_div8)
        self.size16 = (h_div16, w_div16)
        self.size32 = (h_div32, w_div32)

        self.pe = nn.Parameter(torch.zeros(h_div32,  w_div32, channels))
        nn.init.normal_(self.pe, mean=0, std=math.sqrt(1 / channels))

        self.drop = nn.Dropout(drop_prob)
        self.ppm = PyramidPoolingModuleV2(
            in_ch=self.input_channels[-1],
            proj_ch=512,
            out_ch=channels,
            spatial_sizes=(1, 2, 3, 6),
        )

        lion_common_kwargs = dict(attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer)

        self.lion32 = LionLayer(
            hidden_dim=self.hidden_dims[3],
            enc_dim=self.input_channels[3],
            **lion_common_kwargs,
            last_block=False
        )
        self.lion16 = LionLayer(
            hidden_dim=self.hidden_dims[2],
            enc_dim=self.input_channels[2],
            **lion_common_kwargs,
            last_block=False
        )
        self.lion8 = LionLayer(
            hidden_dim=self.hidden_dims[1],
            enc_dim=self.input_channels[1],
            **lion_common_kwargs,
            last_block=False
        )
        self.lion4 = LionLayer(
            hidden_dim=self.hidden_dims[0],
            enc_dim=self.input_channels[0],
            **lion_common_kwargs,
            last_block=True
        )
        final_channels = self.hidden_dims[0] // 2  # [64,]
        self.out_conv = nn.Sequential(
            ConvBN(final_channels, final_channels, 3, use_gn=False, act_layer=act_layer),
            # ConvBN(final_channels, final_channels, 1, use_gn=False, act_layer=act_layer),
            nn.Conv2d(final_channels, output_channel, kernel_size=(1, 1), bias=False)
        )

    def forward(self, features: List[torch.Tensor], input_size: Tuple[int, int]
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        f0, f1, f2, f3 = features
        # f0: (b, 96x288, 192)
        # f1: (b, 48x144, 384)
        # f2: (b, 24x72, 768)
        # f3: (b, 12x36, 1536)

        b = f0.shape[0]  # batch size

        if input_size != self.input_size:
            self.input_size = input_size

            h_div2, w_div2 = input_size[0] // 2, input_size[1] // 2
            h_div4, w_div4 = input_size[0] // 4, input_size[1] // 4
            h_div8, w_div8 = input_size[0] // 8, input_size[1] // 8
            h_div16, w_div16 = input_size[0] // 16, input_size[1] // 16
            h_div32, w_div32 = input_size[0] // 32, input_size[1] // 32

            self.size2 = (h_div2, w_div2)
            self.size4 = (h_div4, w_div4)
            self.size8 = (h_div8, w_div8)
            self.size16 = (h_div16, w_div16)
            self.size32 = (h_div32, w_div32)

        f3 = f3.view(-1, self.size32[0], self.size32[1], self.input_channels[3])
        f2 = f2.view(-1, self.size16[0], self.size16[1], self.input_channels[2])
        f1 = f1.view(-1, self.size8[0], self.size8[1], self.input_channels[1])
        f0 = f0.view(-1, self.size4[0], self.size4[1], self.input_channels[0])

        hidden = f3.permute(0, 3, 1, 2).contiguous()  # (b, 1536, 12, 36)
        hidden = self.ppm(hidden)  # (b, 2048, 12, 36)
        hidden = hidden.permute(0, 2, 3, 1).contiguous()  # (b, 12, 36, 2048)

        pe = self.drop(self.pe)  # (12x36, 2048)
        hidden = hidden + pe

        # -------- 1/32 scale -------- #
        hidden, attn32_1, attn32_2 = self.lion32(hidden, f3)  # (b, 24, 72, 1024)

        # -------- 1/16 scale -------- #
        hidden, attn16_1, attn16_2 = self.lion16(hidden, f2)  # (b, 48, 144, 512)

        # -------- 1/8 scale -------- #
        hidden, attn8_1, attn8_2 = self.lion8(hidden, f1)  # (b, 96, 288, 256)

        # -------- 1/4 scale -------- #
        hidden, attn4_1, attn4_2 = self.lion4(hidden, f0)  # (b, 192, 576, 128) -> (b, 128, 192, 576)

        # -------- 1/2 scale -------- #
        out = self.out_conv(hidden)  # (b, out_ch, 192, 576)

        return out, (attn4_1, attn4_2, attn8_1, attn8_2, attn16_1, attn16_2, attn32_1, attn32_2)
