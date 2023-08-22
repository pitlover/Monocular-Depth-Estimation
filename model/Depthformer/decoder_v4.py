from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from .layer_utils import ResConvBNBlock, ConvBN, ConvBNBlock, UpscaleConcatAct


class DepthFormerDecoderV4(nn.Module):

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
        self.num_heads = num_heads  # 8, currently not used
        self.input_channels = input_channels  # [24, 40, 64, 176, 512]
        self.num_inputs = len(input_channels)  # 5
        self.num_repeat = num_repeat
        num_layers = self.num_inputs  # 5
        self.head_dim = hidden_dim // self.num_heads  # 64

        if feedforward_dim is None:
            feedforward_dim = self.hidden_dim * 2

        assert self.num_inputs == 5  # fixed for EfficientNet, 1/2, 1/4, 1/8, 1/16, 1/32
        self.img_size = img_size

        self.attn_scaler = math.sqrt(1 / self.head_dim)  # 1/8
        self.cls_scaler = math.sqrt(1 / self.hidden_dim)  # 1/512

        self.depth_cls = nn.Parameter(torch.zeros(1, self.hidden_dim))
        nn.init.normal_(self.depth_cls, mean=0, std=self.cls_scaler)

        self.q_projections = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(num_layers)
        ])

        self.k_projections = nn.ModuleList([
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=(1, 1))
            for i in range(num_layers)
        ])

        self.v_projections = nn.ModuleList([
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=(1, 1))
            for i in range(num_layers)
        ])

        self.upscale_layers = nn.ModuleList([
            UpscaleConcatAct(scale_factor=2, act_layer=act_layer) for _ in range(num_layers - 1)
        ])

        self.post_conv_layers = nn.ModuleList([
            ResConvBNBlock(self.input_channels[i] + self.hidden_dim, self.hidden_dim, 3,
                           num_layers=2, act_layer=act_layer)
            for i in range(num_layers - 1)
        ])
        self.post_conv_layers.append(
            ResConvBNBlock(self.input_channels[-1], self.hidden_dim, 3, num_layers=2, act_layer=act_layer)
        )

        self.post_cls_layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(num_layers)])
        self.post_cls_ln = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(num_layers)])
        self.cls_to_weight_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, feedforward_dim),
                nn.Dropout(drop_prob),
                act_layer(),
                nn.Linear(feedforward_dim, self.hidden_dim),
            ) for _ in range(num_layers)  # similar to FF
        ])
        self.post_weight_layers = nn.ModuleList([
            ConvBN(self.hidden_dim, self.hidden_dim, kernel_size=1, act_layer=None, use_residual=False)
            for _ in range(num_layers)
        ])

        self.final_block = nn.Sequential(
            act_layer(),
            ResConvBNBlock(self.hidden_dim, self.hidden_dim, 3, 2, act_layer=act_layer),
            nn.Conv2d(self.hidden_dim, 1, kernel_size=(1, 1)),
            nn.Hardsigmoid()
        )

    def _patch_to_seq_head(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # (b, d, h, w) -> (b, h*w, d)
        b, d, h, w = x.shape
        x = x.view(b, self.num_heads, -1, h, w).permute(0, 1, 3, 4, 2).reshape(b, self.num_heads, h * w, -1)
        return x, (h, w)

    def forward(self, features: Tuple[torch.Tensor, ...]
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        x0, x1, x2, x3, x4 = features
        # x0: (b, ch1, h/2, w/2)  # 24
        # x1: (b, ch2, h/4, w/4)  # 40
        # x2: (b, ch3, h/8, w/8)  # 64
        # x3: (b, ch4, h/16, w/16)  # 176
        # x4: (b, ch5, h/32, w/32)  # 512

        batch_size, _, out_h, out_w = x0.shape

        c4 = self.post_conv_layers[4](x4)  # (b, ch5, h/32, w/32) -> (b, d, h/32, w/32)
        cls = self.depth_cls.expand(batch_size, 1, self.hidden_dim)  # (b, 1, d)
        cls = cls * self.cls_scaler

        # ----------------------------------------------------------------#
        q4 = self.q_projections[4](cls).view(batch_size, self.num_heads, self.head_dim, 1)  # (b, nh, d/h, 1)
        k4 = self.k_projections[4](c4)  # (b, d, h/32, w/32)
        v4 = self.v_projections[4](c4)  # (b, d, h/32, w/32)

        k4, size4 = self._patch_to_seq_head(k4)  # (b, nh, h/32 * w/32, d/h)
        v4s, _ = self._patch_to_seq_head(v4)  # (b, nh, h/32 * w/32, d/h)

        pre4 = torch.matmul(k4, q4)  # (b, nh, h/32 * w/32, 1)
        attn4 = torch.softmax(pre4 * self.attn_scaler, dim=2)
        cls4 = torch.sum(attn4 * v4s, dim=2).view(batch_size, 1, -1)  # (b, nh, d/h) -> (b, 1, d)
        cls = cls + self.post_cls_layers[4](cls4)  # (b, 1, d)
        cls = self.post_cls_ln[4](cls)  # (b, 1, d)

        weight4 = self.cls_to_weight_layers[4](cls)  # (b, 1, d)
        weight4 = weight4.view(batch_size, -1, 1, 1)  # (b, d, 1, 1)
        v4 = v4 * torch.sigmoid(weight4)  # like GLU
        v4 = c4 + self.post_weight_layers[4](v4)

        c3 = self.upscale_layers[3](x3, v4)
        c3 = self.post_conv_layers[3](c3)  # (b, ch4 + d, h/16, w/16) -> (b, d, h/16, w/16)

        # ----------------------------------------------------------------#
        q3 = self.q_projections[3](cls).view(batch_size, self.num_heads, self.head_dim, 1)  # (b, nh, d/h, 1)
        k3 = self.k_projections[3](c3)  # (b, d, h/16, w/16)
        v3 = self.v_projections[3](c3)  # (b, d, h/16, w/16)

        k3, size3 = self._patch_to_seq_head(k3)  # (b, nh, h/16 * w/16, d/h)
        v3s, _ = self._patch_to_seq_head(v3)  # (b, nh, h/16 * w/16, d/h)

        pre3 = torch.matmul(k3, q3)  # (b, nh, h/16 * w/16, 1)
        attn3 = torch.softmax(pre3 * self.attn_scaler, dim=2)
        cls3 = torch.sum(attn3 * v3s, dim=2).view(batch_size, 1, -1)  # (b, nh, d/h) -> (b, 1, d)
        cls = cls + self.post_cls_layers[3](cls3)  # (b, 1, d)
        cls = self.post_cls_ln[3](cls)  # (b, 1, d)

        weight3 = self.cls_to_weight_layers[3](cls)  # (b, 1, d)
        weight3 = weight3.view(batch_size, -1, 1, 1)  # (b, d, 1, 1)
        v3 = v3 * torch.sigmoid(weight3)  # like GLU
        v3 = c3 + self.post_weight_layers[3](v3)

        c2 = self.upscale_layers[2](x2, v3)
        c2 = self.post_conv_layers[2](c2)  # (b, ch3 + d, h/8, w/8) -> (b, d, h/8, w/8)

        # ----------------------------------------------------------------#
        q2 = self.q_projections[2](cls).view(batch_size, self.num_heads, self.head_dim, 1)  # (b, nh, d/h, 1)
        k2 = self.k_projections[2](c2)  # (b, d, h/8, w/8)
        v2 = self.v_projections[2](c2)  # (b, d, h/8, w/8)

        k2, size2 = self._patch_to_seq_head(k2)  # (b, nh, h/8 * w/8, d/h)
        v2s, _ = self._patch_to_seq_head(v2)  # (b, nh, h/8 * w/8, d/h)

        pre2 = torch.matmul(k2, q2)  # (b, nh, h/8 * w/8, 1)
        attn2 = torch.softmax(pre2 * self.attn_scaler, dim=2)
        cls2 = torch.sum(attn2 * v2s, dim=2).view(batch_size, 1, -1)  # (b, nh, d/h) -> (b, 1, d)
        cls = cls + self.post_cls_layers[2](cls2)  # (b, 1, d)
        cls = self.post_cls_ln[2](cls)  # (b, 1, d)

        weight2 = self.cls_to_weight_layers[2](cls)  # (b, 1, d)
        weight2 = weight2.view(batch_size, -1, 1, 1)  # (b, d, 1, 1)
        v2 = v2 * torch.sigmoid(weight2)  # like GLU
        v2 = c2 + self.post_weight_layers[2](v2)

        c1 = self.upscale_layers[1](x1, v2)
        c1 = self.post_conv_layers[1](c1)  # (b, ch2 + d, h/4, w/4) -> (b, d, h/4, w/4)

        # ----------------------------------------------------------------#
        q1 = self.q_projections[1](cls).view(batch_size, self.num_heads, self.head_dim, 1)  # (b, nh, d/h, 1)
        k1 = self.k_projections[1](c1)  # (b, d, h/4, w/4)
        v1 = self.v_projections[1](c1)  # (b, d, h/4, w/4)

        k1, size1 = self._patch_to_seq_head(k1)  # (b, nh, h/4 * w/4, d/h)
        v1s, _ = self._patch_to_seq_head(v1)  # (b, nh, h/4 * w/4, d/h)

        pre1 = torch.matmul(k1, q1)  # (b, nh, h/4 * w/4, 1)
        attn1 = torch.softmax(pre1 * self.attn_scaler, dim=2)
        cls1 = torch.sum(attn1 * v1s, dim=2).view(batch_size, 1, -1)  # (b, nh, d/h) -> (b, 1, d)
        cls = cls + self.post_cls_layers[1](cls1)  # (b, 1, d)
        cls = self.post_cls_ln[1](cls)  # (b, 1, d)

        weight1 = self.cls_to_weight_layers[1](cls)  # (b, 1, d)
        weight1 = weight1.view(batch_size, -1, 1, 1)  # (b, d, 1, 1)
        v1 = v1 * torch.sigmoid(weight1)  # like GLU
        v1 = c1 + self.post_weight_layers[1](v1)

        c0 = self.upscale_layers[0](x0, v1)
        c0 = self.post_conv_layers[0](c0)  # (b, ch2 + d, h/2, w/2) -> (b, d, h/2, w/2)

        # ----------------------------------------------------------------#
        q0 = self.q_projections[0](cls).view(batch_size, self.num_heads, self.head_dim, 1)  # (b, nh, d/h, 1)
        k0 = self.k_projections[0](c0)  # (b, d, h/2, w/2)
        v0 = self.v_projections[0](c0)  # (b, d, h/2, w/2)

        k0, size0 = self._patch_to_seq_head(k0)  # (b, nh, h/2 * w/2, d/h)
        v0s, _ = self._patch_to_seq_head(v0)  # (b, nh, h/2 * w/2, d/h)

        pre0 = torch.matmul(k0, q0)  # (b, nh, h/2 * w/2, 1)
        attn0 = torch.softmax(pre0 * self.attn_scaler, dim=2)
        cls0 = torch.sum(attn0 * v0s, dim=2).view(batch_size, 1, -1)  # (b, nh, d/h) -> (b, 1, d)
        cls = cls + self.post_cls_layers[0](cls0)  # (b, 1, d)
        cls = self.post_cls_ln[0](cls)  # (b, 1, d)

        weight0 = self.cls_to_weight_layers[0](cls)  # (b, 1, d)
        weight0 = weight0.view(batch_size, -1, 1, 1)  # (b, d, 1, 1)
        v0 = v0 * torch.sigmoid(weight0)  # like GLU
        v0 = c0 + self.post_weight_layers[0](v0)

        out = self.final_block(v0)

        # ----------------------------------------------------------------#

        return out, (attn0, attn1, attn2, attn3, attn4)


if __name__ == '__main__':
    dummy1 = torch.zeros(4, 24, 320, 240)
    dummy2 = torch.zeros(4, 40, 160, 120)
    dummy3 = torch.zeros(4, 64, 80, 60)
    dummy4 = torch.zeros(4, 176, 40, 30)
    dummy5 = torch.zeros(4, 512, 20, 15)

    dec = DepthFormerDecoderV4(
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
