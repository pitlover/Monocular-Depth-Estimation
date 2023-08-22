from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .oda2_swin_transformer import SwinTransformer
from .oda2_luna_cls_decoder import ODA2LunaClsDecoder


class ODA2LunaClsModel(nn.Module):

    def __init__(self,
                 decoder_channels: int,
                 num_aux: int,
                 aux_dims: int,
                 num_heads: int,
                 min_depth: float,
                 max_depth: float,
                 encoder_type: str = "large",
                 drop_prob: float = 0.0,
                 attn_drop_prob: float = 0.0,
                 path_drop_prob: float = 0.2,
                 act_layer=nn.GELU,
                 ):
        super().__init__()

        swin_kwargs = dict(pretrain_img_size=224, patch_size=4,
                           depths=(2, 2, 18, 2), window_size=7,
                           drop_prob=drop_prob, attn_drop_prob=attn_drop_prob, path_drop_prob=path_drop_prob)
        if (encoder_type == "base") or (encoder_type == "B"):
            swin = SwinTransformer(embed_dim=128, num_heads=(4, 8, 16, 32), **swin_kwargs)
            swin.init_weights(pretrained="checkpoint/swin_base_patch4_window7_224_22k.pth")
        elif (encoder_type == "large") or (encoder_type == "L"):
            swin = SwinTransformer(embed_dim=192, num_heads=(6, 12, 24, 48), **swin_kwargs)
            swin.init_weights(pretrained="checkpoint/swin_large_patch4_window7_224_22k.pth")
        else:
            raise ValueError(f"Unsupported SwinTransformer type {encoder_type}.")

        self.encoder = swin
        self.decoder = ODA2LunaClsDecoder(
            decoder_channels,
            input_channels=swin.num_features,  # (192, 384, 768, 1536)
            num_aux=num_aux,
            aux_dims=aux_dims,
            num_heads=num_heads,
            attn_drop_prob=attn_drop_prob,
            drop_prob=drop_prob,
            act_layer=act_layer
        )
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        features = self.encoder(x)
        bin_probs, bin_widths = self.decoder(features)
        bin_widths = (self.max_depth - self.min_depth) * bin_widths
        bin_widths = F.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)  # (b, n_bins + 1)
        bin_edges = torch.cumsum(bin_widths, dim=-1)  # (b, n_bins + 1)

        bin_centers = 0.5 * (bin_edges[..., :-1] + bin_edges[..., 1:])  # (b, n_bins)
        bin_centers = bin_centers.unsqueeze(-1).unsqueeze(-1)  # (b, n_bins, 1, 1)

        out = torch.sum(bin_probs * bin_centers, dim=1, keepdim=True)  # (b, 1, h, w)

        return out, bin_centers

    @classmethod
    def build(cls, opt, min_depth: float, max_depth: float):
        # opt = opt["model"]
        m = cls(
            decoder_channels=opt["decoder_channels"],
            num_aux=opt["num_aux"],
            aux_dims=opt["aux_dims"],
            num_heads=opt["num_heads"],
            min_depth=min_depth,
            max_depth=max_depth,
            encoder_type=opt["encoder_type"],
            drop_prob=opt.get("drop_prob", 0.0),
            attn_drop_prob=opt.get("attn_drop_prob", 0.0),
            path_drop_prob=opt.get("path_drop_prob", 0.2),
        )
        print(f"Model built! #params {m.count_params()}")
        return m

    def count_params(self) -> int:
        count = 0
        for p in self.parameters():
            count += p.numel()
        return count


if __name__ == '__main__':
    net = ODA2LunaClsModel(decoder_channels=1536, num_aux=256, aux_dims=256, num_heads=4,
                           min_depth=0.0, max_depth=80.0, encoder_type="L")
    dummy_input = torch.empty(2, 3, 352, 1216)
    dummy_output = net(dummy_input)[0]
    print(dummy_output.shape)  # (2, 1, 176, 608)
