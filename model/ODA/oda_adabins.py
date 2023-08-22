from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from .encoder import ODASwinEncoder
from .decoder_conv import ODAConvDecoder
from ..Adabins.miniViT import mViT


class ODABinsModel(nn.Module):

    def __init__(self,
                 input_size: Tuple[int, int],
                 decoder_channels: int,
                 n_bins: int = 256,
                 min_val: float = 0.1,
                 max_val: float = 10.0,
                 model_name: str = "swin_large_patch4_window12_384_in22k",
                 norm='linear',
                 act_layer=nn.GELU):
        super(ODABinsModel, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.encoder = ODASwinEncoder(input_size, model_name=model_name)
        self.adaptive_bins_layer = mViT(
            in_channels=decoder_channels // 8,
            n_query_channels=128,
            patch_size=16,
            dim_out=n_bins,
            embedding_dim=decoder_channels // 8,
            num_heads=4,
            norm=norm
        )
        self.decoder = ODAConvDecoder(
            decoder_channels,
            input_channels=(192, 384, 768, 1536),
            output_channel=decoder_channels // 8,
            input_size=input_size,
            act_layer=act_layer
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(decoder_channels // 8, n_bins, kernel_size=(1, 1)),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor):
        unet_out = self.decoder(self.encoder(x))

        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
        out = self.conv_out(range_attention_maps)

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        batch_size, num_bins = centers.shape
        centers = centers.view(batch_size, num_bins, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)

        return pred, bin_edges

    def count_params(self) -> int:
        count = 0
        for p in self.parameters():
            count += p.numel()
        return count

    @classmethod
    def build(cls, opt, min_depth: float, max_depth: float):
        # opt = opt["model"]

        m = cls(
            input_size=opt["input_size"],
            n_bins=opt["num_bins"],
            decoder_channels=opt["decoder_channels"],
            min_val=min_depth,
            max_val=max_depth
        )
        print(f"Model built! #params: {m.count_params()}")
        return m


if __name__ == '__main__':
    net = ODABinsModel(input_size=(384, 1152), decoder_channels=1024,
                       n_bins=256, min_val=0.001, max_val=80)
    dummy_input = torch.empty(1, 3, 352, 1216)
    dummy_output = net(dummy_input)
    print(dummy_output.shape)  # (384//2, 1152//2)
