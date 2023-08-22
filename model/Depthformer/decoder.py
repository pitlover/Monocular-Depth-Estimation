from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from .vit_layer import ViTLayer
from .layer_utils import ConvBN, ConvBNBlock, UpscaleConcatAct


class DepthFormerDecoder(nn.Module):

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

        self.hidden_dim = hidden_dim  # 256
        self.num_heads = num_heads  # 4
        self.input_channels = input_channels  # [24, 40, 64, 176, 512]
        self.num_inputs = len(input_channels)  # 5
        self.num_repeat = num_repeat
        num_layers = self.num_inputs  # 5

        assert self.num_inputs == 5  # fixed for EfficientNet, 1/2, 1/4, 1/8, 1/16, 1/32
        # self.img_size = img_size

        self.vit_layers = nn.ModuleList([
            ViTLayer(hidden_dim, num_heads, num_repeat=num_repeat,
                     feedforward_dim=feedforward_dim, attn_drop_prob=attn_drop_prob,
                     drop_prob=drop_prob, act_layer=act_layer)
            for _ in range(num_layers - 1)
        ])

        self.vit_bn_layers = nn.ModuleList([
            nn.BatchNorm2d(hidden_dim, eps=1e-5)
            for _ in range(num_layers - 1)
        ])

        self.post_conv_layers = nn.ModuleList([
            ConvBNBlock(
                in_channels=self.input_channels[i] + hidden_dim if (i != num_layers - 1) else self.input_channels[i],
                out_channels=hidden_dim,
                kernel_size=(2 * (num_layers - i) - 1), num_layers=2, act_layer=act_layer)
            # kernel_size=3, num_layers=2, act_layer=act_layer)
            for i in range(num_layers)
        ])  # 0, 1, 2, 3, 4 -> 9, 7, 5, 3, 1

        self.patchify_layers = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim,
                      kernel_size=(2 ** i, 2 ** i), stride=(2 ** i, 2 ** i), padding=(0, 0))
            for i in range(num_layers - 2, -1, -1)
        ])  # 3, 2, 1, 0 -> 8, 4, 2, 1

        self.upscale_layers = nn.ModuleList([
            UpscaleConcatAct(scale_factor=2 ** (i + 1), act_layer=act_layer)
            for i in range(num_layers - 2, -1, -1)
        ])  # 3, 2, 1, 0 -> 16, 8, 4, 2

        self.final_block = nn.Sequential(
            ConvBN(hidden_dim, hidden_dim // 2, kernel_size=3, act_layer=act_layer),
            ConvBN(hidden_dim // 2, hidden_dim // 4, kernel_size=3, act_layer=act_layer),
            nn.Conv2d(hidden_dim // 4, 1, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

        self.img_size = img_size
        self.num_tokens = (img_size[0] // 32, img_size[1] // 32)
        self.position_embedding = nn.Parameter(
            torch.zeros(self.num_tokens[0] * self.num_tokens[1], self.hidden_dim))  # (s, d)
        nn.init.xavier_normal_(self.position_embedding)

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
        # (b, d, h/32, w/32) -> (b, d, h/32, w/32) because patch=(1, 1)
        patch4 = self.patchify_layers[3](c4)  # (b, d, h/32, w/32)
        patch4, size4 = self._patch_to_seq(patch4)  # (b, s, d)
        patch4 = patch4 + self.position_embedding
        feat4, attn4 = self.vit_layers[3](patch4)  # (b, s, d)
        feat4 = self._seq_to_patch(feat4, size4)  # (b, d, h/32, w/32)
        feat4 = self.vit_bn_layers[3](feat4)

        c3 = self.upscale_layers[3](x3, feat4)
        c3 = self.post_conv_layers[3](c3)  # (b, ch4 + d, h/16, w/16) -> (b, d, h/16, w/16)

        # ----------------------------------------------------------------#
        # (b, d, h/16, w/16) -> (b, d, h/32, w/32) because patch=(2, 2)
        patch3 = self.patchify_layers[2](c3)
        patch3, size3 = self._patch_to_seq(patch3)  # (b, s, d)
        patch3 = patch3 + self.position_embedding
        feat3, attn3 = self.vit_layers[2](patch3)  # (b, s, d)
        feat3 = self._seq_to_patch(feat3, size3)  # (b, d, h/32, w/32)
        feat3 = self.vit_bn_layers[2](feat3)

        c2 = self.upscale_layers[2](x2, feat3)
        c2 = self.post_conv_layers[2](c2)  # (b, ch3 + d, h/8, w/8) -> (b, d, h/8, w/8)

        # ----------------------------------------------------------------#
        # (b, d, h/8, w/8) -> (b, d, h/32, w/32) because patch=(4, 4)
        patch2 = self.patchify_layers[1](c2)
        patch2, size2 = self._patch_to_seq(patch2)  # (b, s, d)
        patch2 = patch2 + self.position_embedding
        feat2, attn2 = self.vit_layers[1](patch2)  # (b, s, d)
        feat2 = self._seq_to_patch(feat2, size2)  # (b, d, h/32, w/32)
        feat2 = self.vit_bn_layers[1](feat2)

        c1 = self.upscale_layers[1](x1, feat2)
        c1 = self.post_conv_layers[1](c1)  # (b, ch2 + d, h/4, w/4) -> (b, d, h/4, w/4)

        # ----------------------------------------------------------------#
        # (b, d, h/4, w/4) -> (b, d, h/32, w/32) because patch=(8, 8)
        patch1 = self.patchify_layers[0](c1)
        patch1, size1 = self._patch_to_seq(patch1)  # (b, s, d)
        patch1 = patch1 + self.position_embedding
        feat1, attn1 = self.vit_layers[0](patch1)  # (b, s, d)
        feat1 = self._seq_to_patch(feat1, size1)  # (b, d, h/32, w/32)
        feat1 = self.vit_bn_layers[0](feat1)

        c0 = self.upscale_layers[0](x0, feat1)
        c0 = self.post_conv_layers[0](c0)  # (b, ch1 + d, h/2, w/2) -> (b, d, h/2, w/2)
        # ----------------------------------------------------------------#
        out = self.final_block(c0)  # (b, 1, h/2, w/2)
        return out, (attn1, attn2, attn3, attn4)


if __name__ == '__main__':
    dummy1 = torch.zeros(4, 24, 320, 240)
    dummy2 = torch.zeros(4, 40, 160, 120)
    dummy3 = torch.zeros(4, 64, 80, 60)
    dummy4 = torch.zeros(4, 176, 40, 30)
    dummy5 = torch.zeros(4, 512, 20, 15)

    dec = DepthFormerDecoder(
        hidden_dim=256,
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
