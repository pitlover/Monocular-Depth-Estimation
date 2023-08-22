from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

from .encoder import ODASwinEncoder
from .decoder_jeju import ODAJejuDecoder
from .layer_utils import ScaledSigmoid


class ODAJejuModel(nn.Module):

    def __init__(self,
                 input_size: Tuple[int, int],  # NOT size of x, but desired input size to the model.
                 decoder_channels: int,  # 2048
                 min_depth: float,
                 max_depth: float,
                 num_aux: int,  # 128
                 aux_dim: int,  # 2048
                 num_heads: int,  # 64
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1,
                 use_gn: bool = False,
                 num_groups: int = 1,
                 model_name: str = "swin_large_patch4_window12_384_in22k",
                 act_layer=nn.GELU,
                 out_func: str = "sigmoid",
                 ):
        super().__init__()
        self.encoder = ODASwinEncoder(input_size, model_name=model_name)
        self.decoder = ODAJejuDecoder(
            decoder_channels,
            input_channels=(192, 384, 768, 1536),
            input_size=input_size,
            num_aux=num_aux,
            aux_dim=aux_dim,
            num_heads=num_heads,
            attn_drop_prob=attn_drop_prob,
            drop_prob=drop_prob,
            output_channel=1,  # regress output
            use_gn=use_gn,
            num_groups=num_groups,
            act_layer=act_layer
        )
        self.min_depth = min_depth
        self.max_depth = max_depth

        out_func = out_func.lower()
        self.out_func_type = out_func
        if out_func == "sigmoid":
            self.out_func = nn.Sigmoid()
        elif out_func == "scaled_sigmoid":
            self.out_func = ScaledSigmoid(is_trainable=False, alpha=4.0, beta=0.5)
        elif out_func == "inv_scaled_sigmoid":
            self.out_func = ScaledSigmoid(is_trainable=False, alpha=0.25, beta=0.5)
        elif out_func == "relu":
            self.out_func = nn.ReLU()
        else:
            raise ValueError(f"Unsupported out_func {out_func}.")

        # for module_name, module in self.encoder.named_modules():
        #     if isinstance(module, nn.Dropout):
        #         if "pos" in module_name:
        #             continue
        #         elif "attn" in module_name:
        #             module.p = attn_drop_prob
        #         else:
        #             module.p = drop_prob

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        orig_h, orig_w = x.shape[-2:]
        new_h = int(np.round(orig_h / 384) * 384)
        new_w = int(np.round(orig_w / 384) * 384)

        features = self.encoder(x)
        out, aux, attn_weights = self.decoder(features, (new_h, new_w))

        out = self.out_func(out)
        if "sigmoid" in self.out_func_type:
            out = out * (self.max_depth - self.min_depth) + self.min_depth
        elif self.out_func_type == "relu":
            out = (out * self.max_depth) + self.min_depth
        else:  # should not be here
            raise ValueError

        return out, aux, attn_weights

    @classmethod
    def build(cls, opt, min_depth: float, max_depth: float):
        # opt = opt["model"]
        m = cls(
            input_size=opt["input_size"],
            decoder_channels=opt["decoder_channels"],
            min_depth=min_depth,
            max_depth=max_depth,
            num_aux=opt["num_aux"],
            aux_dim=opt["aux_dim"],
            num_heads=opt["num_heads"],
            attn_drop_prob=opt.get("attn_drop_prob", 0.0),
            drop_prob=opt.get("drop_prob", 0.1),
            use_gn=opt.get("use_gn", False),
            num_groups=opt.get("num_groups", 1),
            out_func=opt["out_func"],
        )
        print(f"Model built! #params: {m.count_params()}")
        return m

    def count_params(self) -> int:
        count = 0
        for p in self.parameters():
            count += p.numel()
        return count


if __name__ == '__main__':
    net = ODAJejuModel(input_size=(384, 1152), decoder_channels=2048,
                       min_depth=0.001, max_depth=80, num_heads=64, num_aux=128, aux_dim=2048)
    dummy_input = torch.empty(1, 3, 352, 1216)
    dummy_output, dummy_aux, dummy_attn_weights = net(dummy_input)
    print(dummy_output.shape)  # (b, 1, 384//2, 1152//2)
    print(dummy_aux.shape)  # (b, 256, 1024)
