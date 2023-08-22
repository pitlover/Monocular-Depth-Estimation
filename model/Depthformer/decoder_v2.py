from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from .vit_layer import ViTLayer
from .layer_utils import ConvBNBlock, UpscaleConcatAct


class DepthFormerDecoderV2(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 input_channels: Tuple[int, ...],
                 img_size: Tuple[int, int],
                 num_repeat: int = 1,
                 feedforward_dim: Optional[int] = None,
                 attn_drop_prob: float = 0.1,
                 drop_prob: float = 0.1,
                 act_layer=nn.GELU):
        super().__init__()

        self.hidden_dim = hidden_dim  # 512
        self.num_heads = num_heads  # 8
        self.input_channels = input_channels  # [24, 40, 64, 176, 512]
        self.num_inputs = len(input_channels)  # 5
        self.num_repeat = num_repeat
        num_layers = self.num_inputs  # 5

        assert self.num_inputs == 5  # fixed for EfficientNet, 1/2, 1/4, 1/8, 1/16, 1/32
        # self.img_size = img_size

        assert (hidden_dim % 16 == 0) and (num_heads % 4 == 0)

        self.vit_dims = [hidden_dim // 16, hidden_dim // 4, hidden_dim]  # because(->token*4)
        self.vit_heads = [num_heads // 4, num_heads // 2, num_heads]

        self.vit_layers = nn.ModuleList([
            ViTLayer(self.vit_dims[i], self.vit_heads[i], num_repeat=num_repeat, feedforward_dim=feedforward_dim,
                     attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer)
            for i in range(num_layers - 2)
        ])

        self.vit_bn_layers = nn.ModuleList([
            nn.BatchNorm2d(self.vit_dims[i], eps=1e-5)
            for i in range(num_layers - 2)
        ])

        post_conv_layers = []
        for i in range(num_layers):  # 0, 1, 2, 3, 4
            if i == 0:
                in_ch = self.input_channels[i] + self.vit_dims[0]
                out_ch = self.vit_dims[0]
            elif i == 1:
                in_ch = self.input_channels[i] + self.vit_dims[0]
                out_ch = self.vit_dims[0]
            elif i != num_layers - 1:  # 2, 3
                in_ch = self.input_channels[i] + self.vit_dims[i - 1]
                out_ch = self.vit_dims[i - 2]
            else:
                in_ch = self.input_channels[i]
                out_ch = self.vit_dims[-1]
            pc = ConvBNBlock(in_channels=in_ch, out_channels=out_ch,
                             kernel_size=3, num_layers=2, act_layer=act_layer)
            post_conv_layers.append(pc)

        self.post_conv_layers = nn.ModuleList(post_conv_layers)
        # no patchify : we want depth estimation. so we need value each pixel. but use patchify data //32
        self.upscale_layers = nn.ModuleList([
            UpscaleConcatAct(scale_factor=2, act_layer=None)
            for _ in range(num_layers - 1)
        ])

        self.final_block = nn.Sequential(
            nn.Conv2d(self.vit_dims[0], 1, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

        self.img_size = img_size
        position_embeddings = []
        for i in range(num_layers - 2):  # 0, 1, 2
            stride = 2 ** (i + 3)   # 1/8, 1/16, 1/32
            pe_h, pe_w = img_size[0] // stride, img_size[1] // stride
            pe = nn.Parameter(torch.zeros(pe_h * pe_w, self.vit_dims[i]))
            nn.init.xavier_normal_(pe)
            position_embeddings.append(pe)
        self.position_embeddings = nn.ParameterList(position_embeddings)  # position embedding for patch learning

    def _patch_to_seq(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:  # noqa
        # (b, d, h, w) -> (b, h*w, d)
        b, d, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, -1, d)
        return x, (h, w)

    def _seq_to_patch(self, x: torch.Tensor, orig_size: Tuple[int, int]) -> torch.Tensor:  # noqa
        # (b, h*w, d) -> (b, d, h, w)
        h, w = orig_size
        b, _, d = x.shape
        x = x.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, features: Tuple[torch.Tensor, ...]
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        x0, x1, x2, x3, x4 = features
        # x0: (b, ch1, h/2, w/2)  # 24
        # x1: (b, ch2, h/4, w/4)  # 40
        # x2: (b, ch3, h/8, w/8)  # 64
        # x3: (b, ch4, h/16, w/16)  # 176
        # x4: (b, ch5, h/32, w/32)  # 512

        # batch_size = x0.shape[0]

        c4 = self.post_conv_layers[4](x4)  # (b, ch5, h/32, w/32) -> (b, d, h/32, w/32)

        # ----------------------------------------------------------------#
        patch4, size4 = self._patch_to_seq(c4)  # (b, h/32 * w/32, d)
        patch4 = patch4 + self.position_embeddings[2]
        feat4, attn4 = self.vit_layers[2](patch4)  # (b, h/32 * w/32, d)
        feat4 = self._seq_to_patch(feat4, size4)  # (b, d, h/32, w/32)
        feat4 = self.vit_bn_layers[2](feat4)

        c3 = self.upscale_layers[3](x3, feat4)
        c3 = self.post_conv_layers[3](c3)  # (b, ch4 + d, h/16, w/16) -> (b, d/4, h/16, w/16)

        # ----------------------------------------------------------------#
        patch3, size3 = self._patch_to_seq(c3)  # (b, h/16 * w/16, d/4)
        patch3 = patch3 + self.position_embeddings[1]
        feat3, attn3 = self.vit_layers[1](patch3)  # (b, h/16 * w/16, d/4)
        feat3 = self._seq_to_patch(feat3, size3)  # (b, d/4, h/16, w/16)
        feat3 = self.vit_bn_layers[1](feat3)

        c2 = self.upscale_layers[2](x2, feat3)
        c2 = self.post_conv_layers[2](c2)  # (b, ch3 + d/4, h/8, w/8) -> (b, d/16, h/8, w/8)

        # ----------------------------------------------------------------#
        patch2, size2 = self._patch_to_seq(c2)  # (b, h/8 * w/8, d/16)
        patch2 = patch2 + self.position_embeddings[0]
        feat2, attn2 = self.vit_layers[0](patch2)  # (b, h/8 * w/8, d/16)
        feat2 = self._seq_to_patch(feat2, size2)  # (b, d/16, h/8, w/8)
        feat2 = self.vit_bn_layers[0](feat2)

        c1 = self.upscale_layers[1](x1, feat2)
        c1 = self.post_conv_layers[1](c1)  # (b, ch2 + d/16, h/4, w/4) -> (b, d/16, h/4, w/4)

        # ----------------------------------------------------------------#
        c0 = self.upscale_layers[0](x0, c1)
        c0 = self.post_conv_layers[0](c0)  # (b, ch1 + d/16, h/2, w/2) -> (b, d/16, h/2, w/2)

        # ----------------------------------------------------------------#
        out = self.final_block(c0)  # (b, 1, h/2, w/2)

        return out, (attn2, attn3, attn4)


if __name__ == '__main__':
    dummy1 = torch.zeros(4, 24, 320, 240)
    dummy2 = torch.zeros(4, 40, 160, 120)
    dummy3 = torch.zeros(4, 64, 80, 60)
    dummy4 = torch.zeros(4, 176, 40, 30)
    dummy5 = torch.zeros(4, 512, 20, 15)

    dec = DepthFormerDecoderV2(
        hidden_dim=512,
        num_heads=4,
        img_size=(480, 640),
        input_channels=(24, 40, 64, 176, 512),
    )

    dummy_out, dummy_attn_weights = dec((dummy1, dummy2, dummy3, dummy4, dummy5))
    print(dec)
    print("Output shape:", dummy_out.shape)
    for weight in dummy_attn_weights:
        print("Attention weight shape:", weight.shape)
    assert dummy_out.shape == (4, 1, 320, 240)
