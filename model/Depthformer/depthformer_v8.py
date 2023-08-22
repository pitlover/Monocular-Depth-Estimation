import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .decoder_v8 import DepthFormerDecoderV8


class Encoder(nn.Module):
    def __init__(self, backend: nn.Module):
        super(Encoder, self).__init__()
        self.backend = backend

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [x]
        for k, v in self.backend._modules.items():
            v: nn.Module
            if k == 'blocks':
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class DepthformerV8(nn.Module):
    def __init__(self, backend, opt,
                 min_depth: float, max_depth: float):
        super(DepthformerV8, self).__init__()
        self.encoder = Encoder(backend)
        self.decoder = DepthFormerDecoderV8(
            hidden_dim=opt["hidden_dim"],  # 256?
            num_heads=opt["num_heads"],  # 4?
            num_bins=opt["num_bins"],  # 256?
            num_aux=opt["num_aux"],  # 256?
            input_channels=(24, 40, 64, 176, 512),
            img_size=opt["img_size"],
            attn_drop_prob=opt.get("attn_drop_prob", 0.1),
            drop_prob=opt.get("drop_prob", 0.1),
        )

        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Depthformer-V7 forward.

        :param x:       (batch_size, 3, img_h, img_w)
        :return:        (batch_size, 1, img_h/2, img_w/2)
        """
        # if (x.shape[-2] != self.decoder.img_size[0]) or (x.shape[-1] != self.decoder.img_size[1]):
        #     raise ValueError(f"Depthformer require image size should be always consistent to {self.decoder.img_size},"
        #                      f"but the input (h, w) is {tuple(x.shape[-2:])}")

        encoder_feat = self.encoder(x)  # (b, 3, h, w)

        decoder_input = (encoder_feat[4], encoder_feat[5], encoder_feat[6], encoder_feat[8], encoder_feat[10])
        bin_width, bin_cls, attn_weights = self.decoder(decoder_input)

        # log-domain bin estimation
        bin_width = (self.max_depth - self.min_depth) * bin_width
        bin_width = F.pad(bin_width, (1, 0), mode='constant', value=self.min_depth)  # (b, n_bins + 1)
        bin_edges = torch.cumsum(bin_width, dim=-1)  # (b, n_bins + 1)

        centers = 0.5 * (bin_edges[..., :-1] + bin_edges[..., 1:])  # (b, n_bins)
        centers = centers.unsqueeze(-1).unsqueeze(-1)

        # n_bins, out_h, out_w = bin_cls.shape[1:]
        # centers = centers.transpose(-2, -1).reshape(-1, n_bins, out_h // 16, out_w // 16)
        # centers = centers.repeat_interleave(16, dim=-2).repeat_interleave(16, dim=-1)  # (b, n_bins, h/2, w/2)

        depth_output = torch.sum(bin_cls * centers, dim=1, keepdim=True)  # (b, 1, h/2, w/2)

        return depth_output, centers, attn_weights

    def count_params(self) -> int:
        count = 0
        for p in self.parameters():
            count += p.numel()
        return count

    @classmethod
    def build(cls, opt, min_depth: float, max_depth: float):
        # opt = opt["model"]
        basemodel_name = 'tf_efficientnet_b5_ap'

        # print('Loading base model ()...'.format(basemodel_name), end='')
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)

        # Remove last layers
        del basemodel.conv_head  # 11th
        del basemodel.bn2  # 12th
        del basemodel.act2  # 13th
        del basemodel.global_pool
        del basemodel.classifier

        # Building Encoder-Decoder model
        # print('Building Encoder-Decoder model..', end='')
        m = cls(basemodel, opt, min_depth=min_depth, max_depth=max_depth)
        print(f"Model built! #params: {m.count_params()}")
        return m
