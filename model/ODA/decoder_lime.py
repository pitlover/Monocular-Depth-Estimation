from typing import Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_utils import ConvBN
from .lime_layer import LimeLayer


class ODALimeDecoder(nn.Module):
    """Lion decoder stack, regression."""

    def __init__(self,
                 channels: int,  # 256
                 num_layers: int,  # 16
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

        self.stem_conv = nn.Sequential(
            nn.Conv2d(3, channels // 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(channels // 2),
            act_layer(),
            nn.Conv2d(channels // 2, channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

        enc_dim = sum(input_channels)
        enc_channels = 2048
        self.stem_enc = nn.Sequential(
            nn.LayerNorm(enc_dim),
            nn.Linear(enc_dim, enc_channels)
        )

        lime_common_kwargs = dict(attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer)
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            LimeLayer(channels, enc_channels, **lime_common_kwargs)
            for _ in range(num_layers)
        ])

        self.out_conv = nn.Sequential(
            ConvBN(channels, channels, 3, use_gn=False, act_layer=act_layer),
            ConvBN(channels, channels, 3, use_gn=False, act_layer=act_layer),
            nn.Conv2d(channels, output_channel, kernel_size=(1, 1), bias=False)
        )

    def forward(self, img: torch.Tensor,
                features: List[torch.Tensor], input_size: Tuple[int, int]
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        f0, f1, f2, f3 = features
        # f0: (b, 96x288, 192)
        # f1: (b, 48x144, 384)
        # f2: (b, 24x72, 768)
        # f3: (b, 12x36, 1536)

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

        f3 = f3.view(-1, self.size32[0], self.size32[1], self.input_channels[3]).permute(0, 3, 1, 2)
        f2 = f2.view(-1, self.size16[0], self.size16[1], self.input_channels[2]).permute(0, 3, 1, 2)
        f1 = f1.view(-1, self.size8[0], self.size8[1], self.input_channels[1]).permute(0, 3, 1, 2)
        f0 = f0.view(-1, self.size4[0], self.size4[1], self.input_channels[0]).permute(0, 3, 1, 2)

        f3 = F.interpolate(f3, scale_factor=8, mode="nearest")
        f2 = F.interpolate(f2, scale_factor=4, mode="nearest")
        f1 = F.interpolate(f1, scale_factor=2, mode="nearest")

        enc = torch.cat([f0, f1, f2, f3], dim=1)
        b, enc_ch, _, _ = enc.shape
        enc = enc.view(b, enc_ch, -1).transpose(-2, -1).contiguous()
        enc = self.stem_enc(enc)
        enc /= self.num_layers

        # -------- 1/4 scale -------- #
        hidden = self.stem_conv(img)
        if tuple(hidden.shape[-2:]) != (self.size4[0], self.size4[1]):
            hidden = F.interpolate(hidden, size=(self.size4[0], self.size4[1]), mode="bilinear", align_corners=True)

        attn_weights = []
        for i, layer in enumerate(self.layers):
            hidden, attn_w = layer(hidden, enc)
            attn_weights.append(attn_w)

        out = self.out_conv(hidden)  # (b, out_ch, 96, 384)

        return out, tuple(attn_weights)
