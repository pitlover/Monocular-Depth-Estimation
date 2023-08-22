import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa


class ConvBN(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 act_layer=None,
                 use_residual: bool = True):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"ConvBN kernel size should be odd, got {kernel_size}.")

        self.conv = nn.Conv2d(
            in_channels, out_channels, bias=False,
            kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=(kernel_size // 2, kernel_size // 2),
            padding_mode="replicate"
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.act = act_layer() if (act_layer is not None) else nn.Identity()
        self.use_residual = (in_channels == out_channels) and use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.use_residual:
            x = x + identity
        return x


class ConvBNBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 num_layers: int = 2,
                 act_layer=nn.GELU,
                 use_residual: bool = True):
        super().__init__()

        self.num_layers = num_layers

        channels = in_channels
        layers = []
        for i in range(num_layers):
            layers.append(ConvBN(channels, out_channels, kernel_size=kernel_size,
                                 act_layer=act_layer, use_residual=use_residual))
            channels = out_channels

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResConvBNBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 num_layers: int = 2,
                 act_layer=nn.GELU):
        super().__init__()

        self.num_layers = num_layers

        channels = in_channels
        layers = []
        for i in range(num_layers):
            layers.append(ConvBN(channels, out_channels, kernel_size=kernel_size,
                                 act_layer=act_layer if (i != num_layers - 1) else None,
                                 use_residual=False))
            channels = out_channels

        self.layers = nn.ModuleList(layers)

        self.use_residual = (in_channels == out_channels)
        if not self.use_residual:
            self.shortcut = ConvBN(in_channels, out_channels, kernel_size=1, act_layer=None, use_residual=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        for layer in self.layers:
            x = layer(x)
        identity = self.shortcut(identity)
        x = x + identity
        return x


class UpscaleConcatAct(nn.Module):

    def __init__(self,
                 scale_factor: int,
                 act_layer=nn.GELU):
        super().__init__()
        self.scale_factor = scale_factor
        self.act = act_layer() if (act_layer is not None) else nn.Identity()

    def forward(self,
                x_orig_scale: torch.Tensor,
                y_to_upscale: torch.Tensor) -> torch.Tensor:
        y_orig_scale = F.interpolate(
            y_to_upscale,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=True
        )
        out = torch.cat([x_orig_scale, y_orig_scale], dim=1)
        out = self.act(out)
        return out


class GlobalAvgPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.mean(x, dim=[2, 3])  # (b, c, h, w) -> (b, c)
        return y
