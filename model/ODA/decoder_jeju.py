from typing import Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_utils import ConvBN, PyramidPoolingModuleV2, LateralModule
from .jeju_layer import JejuLayer


class ReorderUpsample1d(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.norm = nn.LayerNorm(hidden_dim // 2, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        x = x.view(b, s, 2, -1).reshape(b, 2 * s, -1)
        x = self.fc(x)
        x = self.norm(x)
        return x


class ReorderUpsample2d(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim // 4, hidden_dim // 4)
        self.norm = nn.LayerNorm(hidden_dim // 4, eps=1e-5)

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        b, s, d = x.shape
        h, w = size
        assert (s == h * w) and (d == self.hidden_dim)

        x = x.view(b, s, 4, -1)
        x0 = x[:, :, 0]  # (b, s, d/4)
        x1 = x[:, :, 1]  # (b, s, d/4)
        x2 = x[:, :, 2]  # (b, s, d/4)
        x3 = x[:, :, 3]  # (b, s, d/4)

        y = torch.zeros(b, 2 * h, 2 * w, d // 4, device=x.device, dtype=x.dtype)
        y[:, ::2, ::2] = x0.view(b, h, w, -1)
        y[:, 1::2, ::2] = x1.view(b, h, w, -1)
        y[:, ::2, 1::2] = x2.view(b, h, w, -1)
        y[:, 1::2, 1::2] = x3.view(b, h, w, -1)

        y = y.reshape(b, 4 * s, -1)
        y = self.fc(y)
        y = self.norm(y)
        return y


class SpatialUpsample2d(nn.Module):

    def __init__(self, hidden_dim: int, out_nchw: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_nchw = out_nchw

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.conv = ConvBN(hidden_dim, hidden_dim // 2, 3, act_layer=None)
        self.conv = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), padding_mode="replicate", bias=not self.out_nchw)
        # self.bn = nn.Sequential(
        #     nn.BatchNorm2d(hidden_dim // 2, eps=1e-5),
        #     nn.GELU()
        # )
        if not self.out_nchw:
            self.norm = nn.LayerNorm(hidden_dim // 2, eps=1e-5)
        else:
            self.norm = nn.Sequential(
                nn.BatchNorm2d(hidden_dim // 2, eps=1e-5),
                nn.GELU()
            )

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        # b, s, d = x.shape
        b, d, _, _ = x.shape
        # h, w = size
        # assert (s == h * w) and (d == self.hidden_dim)

        # x = x.transpose(1, 2).reshape(b, d, h, w)
        x = self.up(x)
        x = self.conv(x)

        if not self.out_nchw:
            x = x.view(b, d // 2, -1).transpose(1, 2).contiguous()
        x = self.norm(x)
        return x


class ODAJejuDecoder(nn.Module):
    """Jeju decoder stack, regression."""

    def __init__(self,
                 channels: int,  # 2048
                 input_channels: Tuple[int, int, int, int],
                 input_size: Tuple[int, int],  # same as Encoder
                 num_aux: int,  # 128
                 aux_dim: int,  # 2048
                 num_heads: int,  # 64
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
        # [128, 256, 512, 1024]

        assert aux_dim == channels  # currently required
        # self.hidden_dims = [channels // 64, channels // 16, channels // 4, channels]
        self.hidden_dims = [channels // 8, channels // 4, channels // 2, channels]
        self.num_heads = [num_heads // 8, num_heads // 4, num_heads // 2, num_heads]
        self.aux_dims = [aux_dim // 8, aux_dim // 4, aux_dim // 2, aux_dim]

        self.aux = nn.Parameter(torch.zeros(1, num_aux, aux_dim))
        nn.init.normal_(self.aux, mean=0, std=math.sqrt(1 / aux_dim))

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

        # self.pe = nn.Parameter(torch.zeros(h_div32 * w_div32, channels))
        # nn.init.normal_(self.pe, mean=0, std=math.sqrt(1 / channels))

        self.drop = nn.Dropout(drop_prob)
        self.ppm = PyramidPoolingModuleV2(
            in_ch=self.input_channels[-1],
            proj_ch=512,
            out_ch=channels,
            spatial_sizes=(1, 2, 3, 6),
        )
        # self.lat4 = LateralModule(self.input_channels[0], self.hidden_dims[0])
        # self.lat8 = LateralModule(self.input_channels[1], self.hidden_dims[1])
        # self.lat16 = LateralModule(self.input_channels[2], self.hidden_dims[2])
        # self.lat32 = LateralModule(self.input_channels[3], self.hidden_dims[3])

        jeju_common_kwargs = dict(attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer)

        self.jeju32 = JejuLayer(
            hidden_dim=self.hidden_dims[3],
            enc_dim=self.input_channels[3],
            aux_dim=self.aux_dims[3],
            num_heads=self.num_heads[3],
            **jeju_common_kwargs
        )
        self.aux_32to16 = ReorderUpsample1d(self.aux_dims[3])
        # self.hidden_32to16 = ReorderUpsample2d(self.hidden_dims[3])
        self.hidden_32to16 = SpatialUpsample2d(self.hidden_dims[3])

        self.jeju16 = JejuLayer(
            hidden_dim=self.hidden_dims[2],
            enc_dim=self.input_channels[2],
            aux_dim=self.aux_dims[2],
            num_heads=self.num_heads[2],
            **jeju_common_kwargs
        )
        self.aux_16to8 = ReorderUpsample1d(self.aux_dims[2])
        # self.hidden_16to8 = ReorderUpsample2d(self.hidden_dims[2])
        self.hidden_16to8 = SpatialUpsample2d(self.hidden_dims[2])

        self.jeju8 = JejuLayer(
            hidden_dim=self.hidden_dims[1],
            enc_dim=self.input_channels[1],
            aux_dim=self.aux_dims[1],
            num_heads=self.num_heads[1],
            **jeju_common_kwargs
        )
        self.aux_8to4 = ReorderUpsample1d(self.aux_dims[1])
        # self.hidden_8to4 = ReorderUpsample2d(self.hidden_dims[1])
        self.hidden_8to4 = SpatialUpsample2d(self.hidden_dims[1])

        self.jeju4 = JejuLayer(
            hidden_dim=self.hidden_dims[0],
            enc_dim=self.input_channels[0],
            aux_dim=self.aux_dims[0],
            num_heads=self.num_heads[0],
            **jeju_common_kwargs
        )
        # self.aux_4to2 = ReorderUpsample1d(self.aux_dims[0])
        # self.hidden_4to2 = ReorderUpsample2d(self.hidden_dims[0])
        # final_channels = self.hidden_dims[0] // 4  # [8,]

        # final_channels = self.hidden_dims[0]  # [32,]
        # self.hidden_4to2_norm = nn.LayerNorm(final_channels)
        # self.hidden_4to2 = nn.UpsamplingBilinear2d(scale_factor=2)

        final_channels = self.hidden_dims[0] // 2  # [64,]
        self.hidden_4to2 = SpatialUpsample2d(self.hidden_dims[0], out_nchw=True)
        self.out_conv = nn.Sequential(
            ConvBN(final_channels, final_channels, 3, use_gn=False, act_layer=act_layer),
            ConvBN(final_channels, final_channels, 1, use_gn=False, act_layer=act_layer),
            nn.Conv2d(final_channels, output_channel, kernel_size=(1, 1), bias=False)
        )

        self.norm_f0 = nn.LayerNorm(self.input_channels[0])
        self.norm_f1 = nn.LayerNorm(self.input_channels[1])
        self.norm_f2 = nn.LayerNorm(self.input_channels[2])
        self.norm_f3 = nn.LayerNorm(self.input_channels[3])
        self.norm_ppm = nn.LayerNorm(channels)

    def forward(self, features: List[torch.Tensor], input_size: Tuple[int, int]
                ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        f0, f1, f2, f3 = features
        # f0: (b, 96x288, 192)
        # f1: (b, 48x144, 384)
        # f2: (b, 24x72, 768)
        # f3: (b, 12x36, 1536)
        f0 = self.norm_f0(f0)
        f1 = self.norm_f1(f1)
        f2 = self.norm_f2(f2)
        f3 = self.norm_f3(f3)

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

        # f3 = f3.view(-1, self.size32[0], self.size32[1], self.input_channels[3])
        # f3 = f3.permute(0, 3, 1, 2).contiguous()  # (b, 1536, 12, 36)
        # f2 = f2.view(-1, self.size16[0], self.size16[1], self.input_channels[2])
        # f2 = f2.permute(0, 3, 1, 2).contiguous()  # (b, 768, 24, 72)
        # f1 = f1.view(-1, self.size8[0], self.size8[1], self.input_channels[1])
        # f1 = f1.permute(0, 3, 1, 2).contiguous()  # (b, 384, 48, 144)
        # f0 = f0.view(-1, self.size4[0], self.size4[1], self.input_channels[0])
        # f0 = f0.permute(0, 3, 1, 2).contiguous()  # (b, 1536, 12, 36)

        enc = f3.view(-1, self.size32[0], self.size32[1], self.input_channels[3])
        enc = enc.permute(0, 3, 1, 2).contiguous()  # (b, 1536, 12, 36)
        hidden = self.ppm(enc)  # (b, 2048, 12, 36)
        hidden = hidden.permute(0, 2, 3, 1).reshape(b, -1, self.hidden_dims[3])  # (b, 12x36, 2048)
        hidden = self.norm_ppm(hidden)

        # f3 = self.lat32(f3)  # (b, 1024, 12, 36)
        # f3 = f3.permute(0, 2, 3, 1).reshape(b, -1, self.hidden_dims[3])
        # f2 = self.lat16(f2)  # (b, 512, 24, 72)
        # f2 = f2.permute(0, 2, 3, 1).reshape(b, -1, self.hidden_dims[2])
        # f1 = self.lat8(f1)  # (b, 256, 48, 144)
        # f1 = f1.permute(0, 2, 3, 1).reshape(b, -1, self.hidden_dims[1])
        # f0 = self.lat4(f0)  # (b, 128, 96, 288)
        # f0 = f0.permute(0, 2, 3, 1).reshape(b, -1, self.hidden_dims[0])

        aux = self.drop(self.aux)  # (b, 128, 2048)
        aux = aux * math.sqrt(1 / self.aux_dims[-1])
        # pe = self.drop(self.pe)  # (12x36, 2048)
        # hidden = hidden + pe

        # -------- 1/32 scale -------- #
        hidden, aux, attn32_1, attn32_2 = self.jeju32(hidden, f3, aux, size=self.size32)  # (b, 12x36, 2048)
        hidden = self.hidden_32to16(hidden, self.size32)  # (b, 24x72, 512)
        aux = self.aux_32to16(aux)  # (b, 256, 1024)

        # -------- 1/16 scale -------- #
        hidden, aux, attn16_1, attn16_2 = self.jeju16(hidden, f2, aux, size=self.size16)  # (b, 24x72, 512)
        hidden = self.hidden_16to8(hidden, self.size16)  # (b, 48x144, 128)
        aux = self.aux_16to8(aux)  # (b, 512, 512)

        # -------- 1/8 scale -------- #
        hidden, aux, attn8_1, attn8_2 = self.jeju8(hidden, f1, aux, size=self.size8)  # (b, 48x144, 128)
        hidden = self.hidden_8to4(hidden, self.size8)  # (b, 96x288, 32)
        aux = self.aux_8to4(aux)  # (b, 1024, 256)

        # -------- 1/4 scale -------- #
        hidden, aux, attn4_1, attn4_2 = self.jeju4(hidden, f0, aux, size=self.size4)  # (b, 96x288, 32)
        # b, s, d = hidden.shape
        # hidden = self.hidden_4to2_norm(hidden)
        # hidden = hidden.view(b, self.size4[0], self.size4[1], d).permute(0, 3, 1, 2).contiguous()  # (b, 32, 96, 288)
        hidden = self.hidden_4to2(hidden, self.size4)  # (b, 32, 192, 576)
        # hidden = self.hidden_4to2(hidden, self.size4)  # (b, 32, 192, 576)
        # aux = self.aux_4to2(aux)  # (b, 2048, 128)

        # -------- 1/2 scale -------- #
        out = self.out_conv(hidden)  # (b, out_ch, 192, 576)

        return out, aux, (attn4_1, attn4_2, attn8_1, attn8_2, attn16_1, attn16_2, attn32_1, attn32_2)
