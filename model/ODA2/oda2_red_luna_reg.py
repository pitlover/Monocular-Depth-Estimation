from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .oda2_swin_transformer import SwinTransformer
from .oda2_red_luna_decoder import LunaTransformerRegDecoder


class ODA2RedLunaRegModel(nn.Module):

    def __init__(self,
                 dec_dim: int,  # 512
                 min_depth: float,
                 max_depth: float,
                 num_aux: int,  # 256
                 num_heads: int,  # 8
                 num_layers: int,  # 4
                 encoder_type: str = "large",
                 drop_prob: float = 0.0,
                 attn_drop_prob: float = 0.0,
                 ):
        super().__init__()

        swin_kwargs = dict(pretrain_img_size=224, patch_size=4,
                           depths=(2, 2, 18, 2), window_size=7,
                           # drop_prob=drop_prob, attn_drop_prob=attn_drop_prob, path_drop_prob=path_drop_prob)
                           drop_prob=0.0, attn_drop_prob=0.0, path_drop_prob=0.3)
        if (encoder_type == "base") or (encoder_type == "B"):
            swin = SwinTransformer(embed_dim=128, num_heads=(4, 8, 16, 32), **swin_kwargs)
            swin.init_weights(pretrained="checkpoint/swin_base_patch4_window7_224_22k.pth")
        elif (encoder_type == "large") or (encoder_type == "L"):
            swin = SwinTransformer(embed_dim=192, num_heads=(6, 12, 24, 48), **swin_kwargs)
            swin.init_weights(pretrained="checkpoint/swin_large_patch4_window7_224_22k.pth")
        else:
            raise ValueError(f"Unsupported SwinTransformer type {encoder_type}.")

        self.encoder = swin
        self.decoder = LunaTransformerRegDecoder(
            dec_dim,
            enc_dims=swin.num_features,  # (192, 384, 768, 1536)
            num_aux=num_aux,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_drop_prob=attn_drop_prob,
            drop_prob=drop_prob,
        )
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        features = self.encoder(x)
        out, aux, attn_weights = self.decoder(features)
        out = out * (self.max_depth - self.min_depth) + self.min_depth
        return out, aux, attn_weights

    @classmethod
    def build(cls, opt, min_depth: float, max_depth: float):
        # opt = opt["model"]
        m = cls(
            dec_dim=opt["dec_dim"],
            num_aux=opt["num_aux"],
            num_heads=opt["num_heads"],
            num_layers=opt["num_layers"],
            min_depth=min_depth,
            max_depth=max_depth,
            encoder_type=opt["encoder_type"],
            drop_prob=opt.get("drop_prob", 0.0),
            attn_drop_prob=opt.get("attn_drop_prob", 0.0),
        )
        print(f"Model built! #params {m.count_params()}")
        return m

    def count_params(self) -> int:
        count = 0
        for p in self.parameters():
            count += p.numel()
        return count


if __name__ == '__main__':
    net = ODA2RedLunaRegModel(dec_dim=512, num_heads=8, min_depth=0.0, max_depth=80.0, encoder_type="L",
                              num_aux=256, num_layers=4)
    dummy_input = torch.empty(2, 3, 352, 1216)
    dummy_output = net(dummy_input)[0]
    print(dummy_output.shape)  # (2, 1, 88-2, 304-2)
