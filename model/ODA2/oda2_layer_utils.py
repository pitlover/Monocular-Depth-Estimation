from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

# _CONV_PADDING_MODE = "zeros"


_CONV_PADDING_MODE = "replicate"


class ConvBN(nn.Module):

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int,
                 conv_groups: int = 1,
                 use_gn: bool = False,
                 gn_groups: int = 1,
                 gn_per_group: int = -1,
                 act_layer: Optional = nn.GELU,
                 **act_kwargs):
        super().__init__()
        assert kernel_size % 2 == 1
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=(kernel_size, kernel_size),
            stride=(1, 1),
            padding=(kernel_size // 2, kernel_size // 2),
            padding_mode=_CONV_PADDING_MODE,
            groups=conv_groups,
            bias=False,
        )

        if (gn_per_group > 0) and use_gn:
            if out_ch % gn_per_group != 0:
                raise ValueError(f"GroupNorm ch {out_ch} not divisible by {gn_per_group}.")
            gn_groups = out_ch // gn_per_group

        if not use_gn:
            self.bn = nn.BatchNorm2d(out_ch)
        else:
            self.bn = nn.GroupNorm(gn_groups, out_ch)
        self.act = act_layer(**act_kwargs) if (act_layer is not None) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class PyramidPoolingModule(nn.Module):

    def __init__(self,
                 in_ch: int,  # 1536
                 proj_ch: int,  # 512
                 out_ch: int,  # 2048
                 spatial_sizes: Tuple[int, ...],  # 1, 2, 3, 6
                 act_layer=nn.GELU,
                 **act_kwargs) -> None:
        super().__init__()

        self.in_ch = in_ch
        self.proj_ch = proj_ch
        self.out_ch = out_ch
        self.num_pooling = len(spatial_sizes)

        self.pooling_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d(output_size=(spatial_sizes[i], spatial_sizes[i]))
            for i in range(self.num_pooling)
        ])

        self.conv_reduce_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, proj_ch, (1, 1), bias=False),
                # nn.GroupNorm(8, proj_ch),
                nn.BatchNorm2d(proj_ch),
                # nn.ReLU(),
                act_layer()
            )
            for _ in range(self.num_pooling)
        ])

        total_ch = in_ch + (proj_ch * self.num_pooling)
        self.conv = nn.Sequential(
            nn.Conv2d(total_ch, out_ch, (3, 3), padding=(1, 1), padding_mode=_CONV_PADDING_MODE, bias=False),
            # nn.GroupNorm(8, out_ch),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU()
            act_layer()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, ch, h, w = x.shape
        assert ch == self.in_ch

        spp = [x]
        for (pool, red) in zip(self.pooling_layers, self.conv_reduce_layers):
            pooled_x = pool(x)  # (b, ch, 1, 1)
            reduced_x = red(pooled_x)  # (b, 512, 1, 1)
            upsample_x = F.interpolate(reduced_x, size=(h, w), mode="bilinear", align_corners=True)  # (b, 512, h, w)
            spp.append(upsample_x)

        spp = torch.cat(spp, dim=1)  # (b, ch + 512 * 4, h, w)
        spp = self.conv(spp)
        return spp
