from typing import Tuple
import torch
import torch.nn as nn

from .encoder import ODASwinEncoder
from .decoder_conv import ODAConvDecoder


class ODAConvModel(nn.Module):

    def __init__(self,
                 input_size: Tuple[int, int],  # NOT size of x, but desired input size to the model.
                 decoder_channels: int,
                 min_depth: float,
                 max_depth: float,
                 model_name: str = "swin_large_patch4_window12_384_in22k",
                 act_layer=nn.GELU,
                 ):
        super().__init__()
        self.encoder = ODASwinEncoder(input_size, model_name=model_name)
        self.decoder = ODAConvDecoder(
            decoder_channels,
            input_channels=(192, 384, 768, 1536),
            input_size=input_size,
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
            input_size=opt["input_size"],
            decoder_channels=opt["decoder_channels"],
            min_depth=min_depth,
            max_depth=max_depth
        )
        print(f"Model built! #params: {m.count_params()}")
        return m

    def count_params(self) -> int:
        count = 0
        for p in self.parameters():
            count += p.numel()
        return count


if __name__ == '__main__':
    net = ODAConvModel(input_size=(384, 1152), decoder_channels=1024)
    dummy_input = torch.empty(1, 3, 352, 1216)
    dummy_output = net(dummy_input)
    print(dummy_output.shape)  # (384//2, 1152//2)
