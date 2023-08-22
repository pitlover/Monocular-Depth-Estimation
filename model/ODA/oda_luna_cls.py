from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .encoder import ODASwinEncoder
from .decoder_luna import ODALunaDecoder
from .decoder_luna_rp import ODALunaDecoderRP


class ODALunaClsModel(nn.Module):

    def __init__(self,
                 input_size: Tuple[int, int],  # NOT size of x, but desired input size to the model.
                 decoder_channels: int,  # 1024
                 min_depth: float,
                 max_depth: float,
                 num_bins: int,  # 256
                 num_aux: int,  # 256
                 aux_dim: int,  # 256
                 num_heads: int,  # 8
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1,
                 use_gn: bool = False,
                 num_groups: int = 1,
                 model_name: str = "swin_large_patch4_window12_384_in22k",
                 use_rp:bool=False,
                 act_layer=nn.GELU,
                 ):
        super().__init__()
        self.encoder = ODASwinEncoder(input_size, model_name=model_name)
        if not use_rp:
            self.decoder = ODALunaDecoder(
                decoder_channels,
                input_channels=(192, 384, 768, 1536),
                input_size=input_size,
                num_aux=num_aux,
                aux_dim=aux_dim,
                num_heads=num_heads,
                attn_drop_prob=attn_drop_prob,
                drop_prob=drop_prob,
                output_channel=num_bins,
                use_gn=use_gn,
                num_groups=num_groups,
                act_layer=act_layer
            )
        else:
            self.decoder = ODALunaDecoderRP(
                decoder_channels,
                input_channels=(192, 384, 768, 1536),
                input_size=input_size,
                num_aux=num_aux,
                aux_dim=aux_dim,
                num_heads=num_heads,
                attn_drop_prob=attn_drop_prob,
                drop_prob=drop_prob,
                output_channel=num_bins,
                use_gn=use_gn,
                num_groups=num_groups,
                act_layer=act_layer
            )
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.bin_regressor = nn.Sequential(
            nn.Linear(aux_dim, aux_dim),
            act_layer(),
            nn.Linear(aux_dim, aux_dim),
            act_layer(),
            nn.Linear(aux_dim, num_bins),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        features = self.encoder(x)
        out, aux, attn_weights = self.decoder(features)
        bin_cls = torch.softmax(out, dim=1)  # (b, 256, h, w)

        cls = torch.mean(aux, dim=1)  # (b, n_bins, d) -> (b, d)  # TODO maybe other way to select cls?
        bin_widths = self.bin_regressor(cls)
        bin_widths = F.elu(bin_widths, alpha=0.1)  # TODO maybe ReLU is better?
        bin_widths = bin_widths / torch.sum(bin_widths, dim=-1, keepdim=True)
        bin_widths = (self.max_depth - self.min_depth) * bin_widths
        bin_widths = F.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)  # (b, n_bins + 1)
        bin_edges = torch.cumsum(bin_widths, dim=-1)  # (b, n_bins + 1)

        centers = 0.5 * (bin_edges[..., :-1] + bin_edges[..., 1:])  # (b, n_bins)
        centers = centers.unsqueeze(-1).unsqueeze(-1)

        out = torch.sum(bin_cls * centers, dim=1, keepdim=True)  # (b, 1, h/2, w/2)
        return out, aux, centers, attn_weights

    @classmethod
    def build(cls, opt, min_depth: float, max_depth: float):
        # opt = opt["model"]
        m = cls(
            input_size=opt["input_size"],
            decoder_channels=opt["decoder_channels"],
            min_depth=min_depth,
            max_depth=max_depth,
            num_bins=opt["num_bins"],
            num_aux=opt["num_aux"],
            aux_dim=opt["aux_dim"],
            num_heads=opt["num_heads"],
            attn_drop_prob=opt.get("attn_drop_prob", 0.0),
            drop_prob=opt.get("drop_prob", 0.1),
            use_gn=opt.get("use_gn", False),
            num_groups=opt.get("num_groups", 1),
            use_rp = opt.get("use_rp", False),
        )
        print(f"Model built! #params: {m.count_params()}")
        return m

    def count_params(self) -> int:
        count = 0
        for p in self.parameters():
            count += p.numel()
        return count


if __name__ == '__main__':
    net = ODALunaClsModel(input_size=(384, 1152), decoder_channels=1024, num_bins=256,
                          min_depth=0.001, max_depth=80, num_heads=8, num_aux=256, aux_dim=256)
    dummy_input = torch.empty(1, 3, 352, 1216)
    dummy_output, dummy_aux, dummy_attn_weights = net(dummy_input)
    print(dummy_output.shape)  # (b, 1, 384//2, 1152//2)
    print(dummy_aux.shape)  # (b, 256, 1024)
