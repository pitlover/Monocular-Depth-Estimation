from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .oda2_swin_transformer import SwinTransformer
from .oda2_ksa_decoder import KSATransformerRegDecoder


class ODA2KSARegModel(nn.Module):

    def __init__(self,
                 dec_dim: int,
                 min_depth: float,
                 max_depth: float,
                 depths: Tuple[int, ...] = (2, 2, 2, 2),
                 num_heads: Tuple[int, ...] = (4, 8, 16, 32),
                 window_size: int = 7,
                 mlp_ratio: float = 4.0,
                 encoder_type: str = "large",
                 drop_prob: float = 0.0,
                 attn_drop_prob: float = 0.0,
                 path_drop_prob: float = 0.2,
                 ):
        super().__init__()

        swin_kwargs = dict(pretrain_img_size=224, patch_size=4,
                           depths=(2, 2, 18, 2), window_size=7,
                           # drop_prob=drop_prob, attn_drop_prob=attn_drop_prob, path_drop_prob=path_drop_prob)
                           drop_prob=0.0, attn_drop_prob=0.0, path_drop_prob=0.2)
        if (encoder_type == "base") or (encoder_type == "B"):
            swin = SwinTransformer(embed_dim=128, num_heads=(4, 8, 16, 32), **swin_kwargs)
            swin.init_weights(pretrained="checkpoint/swin_base_patch4_window7_224_22k.pth")
        elif (encoder_type == "large") or (encoder_type == "L"):
            swin = SwinTransformer(embed_dim=192, num_heads=(6, 12, 24, 48), **swin_kwargs)
            swin.init_weights(pretrained="checkpoint/swin_large_patch4_window7_224_22k.pth")
        else:
            raise ValueError(f"Unsupported SwinTransformer type {encoder_type}.")

        self.encoder = swin
        self.decoder = KSATransformerRegDecoder(
            dec_dim,
            enc_dims=swin.num_features,  # (192, 384, 768, 1536)
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            attn_drop_prob=attn_drop_prob,
            drop_prob=drop_prob,
            path_drop_prob=path_drop_prob,
        )
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        features = self.encoder(x)
        out = self.decoder(features)
        out = out * (self.max_depth - self.min_depth) + self.min_depth
        return out, None

    @classmethod
    def build(cls, opt, min_depth: float, max_depth: float):
        # opt = opt["model"]
        m = cls(
            dec_dim=opt["dec_dim"],
            depths=opt["depths"],
            num_heads=opt["num_heads"],
            window_size=opt["window_size"],
            mlp_ratio=opt.get("mlp_ratio", 4.0),
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
    net = ODA2KSARegModel(dec_dim=1024, depths=(2, 2, 2, 2), num_heads=(4, 8, 16, 32),
                          min_depth=0.0, max_depth=80.0, encoder_type="L")
    dummy_input = torch.empty(2, 3, 352, 1216)
    dummy_output = net(dummy_input)[0]
    print(dummy_output.shape)  # (2, 1, 88-2, 304-2)
