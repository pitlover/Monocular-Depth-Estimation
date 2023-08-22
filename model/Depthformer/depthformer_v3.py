from typing import List, Tuple
import torch
import torch.nn as nn

from .decoder_v3 import DepthFormerDecoderV3


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


class DepthformerV3(nn.Module):
    def __init__(self, backend, opt,
                 min_depth: float, max_depth: float):
        super(DepthformerV3, self).__init__()
        self.encoder = Encoder(backend)
        self.decoder = DepthFormerDecoderV3(
            hidden_dim=opt["hidden_dim"],  # 512?
            num_heads=opt["num_heads"],  # 4?
            input_channels=(24, 40, 64, 176, 512),
            img_size=opt["img_size"],
            num_repeat=opt.get("num_repeat", 1),
            attn_drop_prob=opt.get("attn_drop_prob", 0.1),
            drop_prob=opt.get("drop_prob", 0.1),
        )

        self.min_depth = min_depth
        self.max_depth = max_depth

        self.conv_out = nn.Sequential(nn.Conv2d(128, 100, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,Tuple[torch.Tensor, ...]]:
        """Depthformer forward.

        :param x:       (batch_size, 3, img_h, img_w)
        :return:        (batch_size, 1, img_h/2, img_w/2)
                        (batch_size, num_head, num_patch, num_patch) x 4
        """
        if (x.shape[-2] != self.decoder.img_size[0]) or (x.shape[-1] != self.decoder.img_size[1]):
            raise ValueError(f"Depthformer require image size should be always consistent to {self.decoder.img_size},"
                             f"but the input (h, w) is {tuple(x.shape[-2:])}")

        encoder_feat = self.encoder(x)  # (b, 3, h, w)

        decoder_input = (encoder_feat[4], encoder_feat[5], encoder_feat[6], encoder_feat[8], encoder_feat[10])
        pred, range_attn_maps = self.decoder(decoder_input)
        # depth_output pass through sigmoid [0, 1], so we rescale to real depth
        ## jiizero
        out = self.conv_out(range_attn_maps)
        depth_output = (self.max_depth - self.min_depth) * depth_output
        bin_widths = nn.functional.pad(depth_output, (1, 0), mode='constant', value=self.min_depth)
        bin_edges = torch.cumsum(bin_widths, dim = 1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)

        return pred, bin_edges, range_attn_maps

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


if __name__ == '__main__':
    dummy_x = torch.rand(2, 3, 480, 640)
    net = DepthformerV3.build(opt={"hidden_dim": 512, "num_heads": 4, "img_size": (480, 640)},
                              min_depth=0.001, max_depth=80.0)
    depth_pred, bin_edges, attn_weights = net(dummy_x)
    print(depth_pred.shape)
