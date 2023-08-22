from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa


class ConvBN(nn.Module):

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int,
                 use_gn: bool = False,
                 num_groups: int = 1,
                 act_layer: Optional = nn.GELU):
        super().__init__()
        assert kernel_size % 2 == 1
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=(kernel_size, kernel_size),
            stride=(1, 1),
            padding=(kernel_size // 2, kernel_size // 2),
            padding_mode="replicate",
            bias=False,
        )
        if not use_gn:
            self.bn = nn.BatchNorm2d(out_ch)
        else:
            self.bn = nn.GroupNorm(num_groups, out_ch)
        self.act = act_layer() if (act_layer is not None) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ScaledSigmoid(nn.Module):

    def __init__(self, is_trainable: bool = False, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        if is_trainable:
            self.alpha = nn.Parameter(torch.tensor(alpha, ))
            self.beta = nn.Parameter(torch.tensor(beta, ))
        else:
            self.alpha = alpha
            self.beta = beta
        self.is_trainable = is_trainable

    def forward(self, x: torch.tensor) -> torch.Tensor:
        if self.is_trainable:
            alpha = torch.clamp_min(self.alpha, 1.0)
            beta = torch.clamp_min(self.beta, 1.0)
        else:
            alpha = self.alpha
            beta = self.beta

        exp_x = alpha * torch.exp(-x / beta)
        out = 1.0 / (1 + exp_x)
        return out


class PyramidPoolingModule(nn.Module):

    def __init__(self, in_ch: int, out_ch: int,
                 spatial_sizes: Tuple[int, ...]  # 1, 2, 3, 6
                 ):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_pooling = len(spatial_sizes)

        if in_ch % self.num_pooling != 0:
            raise ValueError("In_ch should be divisible by num_pooling.")

        self.pooling_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d(output_size=(spatial_sizes[i], spatial_sizes[i]))
            for i in range(self.num_pooling)
        ])

        self.conv_reduce_layers = nn.ModuleList([
            nn.Conv2d(in_ch, in_ch // self.num_pooling, (1, 1))
            for _ in range(self.num_pooling)
        ])

        # TODO do we need BN?
        self.conv = nn.Conv2d(in_ch * 2, out_ch, (1, 1), bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, ch, h, w = x.shape
        assert ch == self.in_ch

        spp = [x]
        for (pool, red) in zip(self.pooling_layers, self.conv_reduce_layers):
            pooled_x = pool(x)  # (b, ch, 1, 1)
            reduced_x = red(pooled_x)  # (b, ch/4, 1, 1)
            upsampled_x = F.upsample_bilinear(reduced_x, size=(h, w))  # (b, ch/4, h, w)
            spp.append(upsampled_x)

        spp = torch.cat(spp, dim=1)  # (b, ch * 2, h, w)
        spp = self.conv(spp)
        spp = self.bn(spp)
        return spp


class PyramidPoolingModuleV2(nn.Module):

    def __init__(self,
                 in_ch: int,  # 1536
                 proj_ch: int,  # 512
                 out_ch: int,  # 2048
                 spatial_sizes: Tuple[int, ...]  # 1, 2, 3, 6
                 ):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_pooling = len(spatial_sizes)

        self.pooling_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d(output_size=(spatial_sizes[i], spatial_sizes[i]))
            for i in range(self.num_pooling)
        ])

        self.conv_reduce_layers = nn.ModuleList([
            nn.Conv2d(in_ch, proj_ch, (1, 1), bias=False)
            for _ in range(self.num_pooling)
        ])

        total_ch = in_ch + (proj_ch * self.num_pooling)
        self.bn = nn.BatchNorm2d(total_ch)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(total_ch, out_ch, kernel_size=(3, 3),
                              stride=(1, 1), padding=(1, 1), padding_mode="replicate")
        # self.conv = ConvBN(total_ch, out_ch, 3, use_gn=False, act_layer=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, ch, h, w = x.shape
        assert ch == self.in_ch

        spp = [x]
        for (pool, red) in zip(self.pooling_layers, self.conv_reduce_layers):
            pooled_x = pool(x)  # (b, ch, 1, 1)
            reduced_x = red(pooled_x)  # (b, 512, 1, 1)
            upsampled_x = F.interpolate(reduced_x, size=(h, w), mode="bilinear", align_corners=True)  # (b, 512, h, w)
            spp.append(upsampled_x)

        spp = torch.cat(spp, dim=1)  # (b, ch + 512 * 4, h, w)
        spp = self.bn(spp)
        spp = self.act(spp)
        spp = self.conv(spp)
        return spp


class LateralModule(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = ConvBN(in_ch, out_ch, 3, use_gn=False, act_layer=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, ch, h, w = x.shape
        assert ch == self.in_ch
        out = self.conv(x)
        return out
