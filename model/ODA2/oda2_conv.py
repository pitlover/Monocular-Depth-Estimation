from typing import Tuple
import torch
import torch.nn as nn

from .oda2_swin_transformer import SwinTransformer
from .oda2_conv_decoder import ODA2ConvDecoder


class ODA2ConvModel(nn.Module):

    def __init__(self,
                 decoder_channels: int,
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
        self.decoder = ODA2ConvDecoder(
            decoder_channels,
            input_channels=swin.num_features,  # (192, 384, 768, 1536)
            output_channel=1,
            act_layer=act_layer
        )
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        features = self.encoder(x)
        out = self.decoder(features)
        out = torch.sigmoid(out)
        out = out * (self.max_depth - self.min_depth) + self.min_depth
        return out, None

    @classmethod
    def build(cls, opt, min_depth: float, max_depth: float):
        # opt = opt["model"]
        m = cls(
            decoder_channels=opt["decoder_channels"],
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
    net = ODA2ConvModel(decoder_channels=1024, min_depth=0.0, max_depth=80.0, encoder_type="L")
    dummy_input = torch.empty(1, 3, 352, 1216)
    dummy_output = net(dummy_input)[0]
    print(dummy_output.shape)  # (1, 1, 176, 608)
