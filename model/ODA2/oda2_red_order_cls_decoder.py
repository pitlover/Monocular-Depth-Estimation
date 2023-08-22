from typing import Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .oda2_layer_utils import ConvBN, _CONV_PADDING_MODE
from .oda2_red_order_reg_decoder import OrderedReductionBlock


class OrderedReductionClsHead(nn.Module):

    def __init__(self,
                 in_dims: int,
                 num_heads: int,
                 num_repeats: int,
                 num_emb: int = 128,  # num_classes
                 reduction_ratio: int = 8,
                 temperature: float = 1.0,
                 feedforward_dims: Optional[int] = None,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0,
                 act_layer=nn.GELU):
        super().__init__()
        self.in_dims = in_dims
        self.num_repeats = num_repeats
        self.num_emb = num_emb

        conv_kwargs = dict(act_layer=act_layer, use_gn=False)
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                ConvBN(in_dims, in_dims // 4, 3, **conv_kwargs),
                ConvBN(in_dims // 4, in_dims // 4, 3, **conv_kwargs),
                nn.Conv2d(in_dims // 4, num_emb, kernel_size=(1, 1), stride=(1, 1), bias=True)
                # nn.Conv2d(in_dims // 4, 1, (3, 3), stride=(1, 1), padding=(1, 1), padding_mode=_CONV_PADDING_MODE),
            ) for _ in range(num_repeats + 1)
        ])

        self.attn_layers = nn.ModuleList([
            OrderedReductionBlock(in_dims, num_heads, reduction_ratio, act_layer=act_layer,
                                  feedforward_dims=feedforward_dims, attn_drop_prob=attn_drop_prob, drop_prob=drop_prob)
            for _ in range(num_repeats)
        ])
        self.softmax = nn.Softmax(dim=1)
        self.temperature = temperature

        with torch.no_grad():
            # log-scale fixed bins
            bins = np.linspace(-10.0, 0.0, num_emb - 1)[:-1]  # (ch,) [-10.0, ... , 0.0] -> [-10, ...., ]
            bins = np.exp(bins).tolist()
            bins = [0.001] + bins + [0.999]

            # linear-scale fixed bins
            # bins = np.linspace(0.001, 0.999, num_emb)
            bins = torch.tensor(bins, dtype=torch.float32)
            bins = bins.view(1, num_emb, 1, 1)
        self.depth_bins = nn.Parameter(bins, requires_grad=True)
        # self.register_buffer("depth_bins", bins)  # (ch,)

        # initialize depth embedding with sinusoidal
        with torch.no_grad():
            emb = torch.zeros(num_emb, in_dims, dtype=torch.float32)  # (ch, d)
            pos = torch.arange(0, num_emb, dtype=torch.float32)
            # inv_freq = 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
            inv_freq = torch.exp(
                torch.arange(0.0, in_dims, 2.0, dtype=torch.float32).mul_(-math.log(1000.0) / in_dims))
            pos_dot = torch.outer(pos, inv_freq)  # (n,) x (d//2,) -> (n, d//2)
            emb[:, 0::2] = torch.sin(pos_dot)
            emb[:, 1::2] = torch.cos(pos_dot)
            emb *= math.sqrt(1 / in_dims)
        self.depth_embedding = nn.Parameter(emb, requires_grad=True)
        # self.register_buffer("depth_embedding", emb)

    def forward(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        # b, h, w, c = x.shape

        outs = list()
        attn_weights = tuple()

        for i in range(self.num_repeats):
            x_chw = x.permute(0, 3, 1, 2)
            logit = self.conv_layers[i](x_chw)  # (b, ch, h, w)
            prob = self.softmax(logit / self.temperature)  # (b, ch, h, w)  small temperature T = 1/4
            out = torch.sum(prob * self.depth_bins, dim=1, keepdim=True)  # (b, 1, h, w)
            outs.append(out)

            b, ch, h, w = prob.shape
            prob_hwc = prob.permute(0, 2, 3, 1).reshape(-1, ch)  # (bhw, ch)
            de = torch.matmul(prob_hwc, self.depth_embedding)  # (bhw, ch) x (ch, d) = (bhw, d)
            de = de.view(b, h, w, -1)
            # de = self.drop(de)

            x, aws = self.attn_layers[i](x, de)
            attn_weights += aws

        x_chw = x.permute(0, 3, 1, 2)
        logit = self.conv_layers[self.num_repeats](x_chw)  # (b, ch, h, w)
        prob = self.softmax(logit / self.temperature)  # (b, ch, h, w)  small temperature T = 1/4
        out = torch.sum(prob * self.depth_bins, dim=1, keepdim=True)  # (b, 1, h, w)
        outs.append(out)

        outs = tuple(outs)
        return outs, attn_weights


class OrderedReductionClsDecoder(nn.Module):

    def __init__(self,
                 dec_dim: int = 512,
                 enc_dims: Tuple[int, int, int, int] = (192, 384, 768, 1536),
                 num_heads: int = 8,
                 num_repeats: int = 3,
                 num_emb: int = 128,
                 reduction_ratio: int = 8,
                 temperature: float = 1.0,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0,
                 act_layer=nn.GELU):
        super().__init__()

        self.dec_dim = dec_dim
        self.enc_dims = enc_dims
        assert len(enc_dims) == 4
        if dec_dim % 4 != 0:
            raise ValueError(f"Decoder dim {dec_dim} should be a multiple of 4.")

        # -------------------------------------------------------------- #
        # Neck
        # -------------------------------------------------------------- #

        conv_kwargs = dict(act_layer=act_layer, use_gn=False)
        self.enc_conv32 = nn.Sequential(
            ConvBN(enc_dims[3], enc_dims[3], 3, **conv_kwargs),
            ConvBN(enc_dims[3], dec_dim // 4, 3, **conv_kwargs),
            nn.UpsamplingBilinear2d(scale_factor=8)
        )
        self.enc_conv16 = nn.Sequential(
            ConvBN(enc_dims[2], enc_dims[2], 3, **conv_kwargs),
            ConvBN(enc_dims[2], dec_dim // 2, 3, **conv_kwargs),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.enc_conv8 = nn.Sequential(
            ConvBN(enc_dims[1], enc_dims[1], 3, **conv_kwargs),
            ConvBN(enc_dims[1], dec_dim, 3, **conv_kwargs),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.enc_conv4 = nn.Sequential(
            ConvBN(enc_dims[0], enc_dims[0], 3, **conv_kwargs),
            ConvBN(enc_dims[0], dec_dim * 2, 3, **conv_kwargs),
            nn.Identity()  # for consistency
        )

        enc_channels = (dec_dim // 4) + (dec_dim // 2) + dec_dim + (dec_dim * 2)
        self.dec_linear = nn.Linear(enc_channels, dec_dim, bias=False)
        self.dec_norm = nn.LayerNorm(dec_dim, elementwise_affine=True)

        # -------------------------------------------------------------- #
        # Head
        # -------------------------------------------------------------- #

        self.reducer = OrderedReductionClsHead(
            dec_dim, num_heads, num_repeats, num_emb=num_emb, reduction_ratio=reduction_ratio, temperature=temperature,
            attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        for module in self.modules():
            # zero filling biases
            if isinstance(module, nn.Linear) and (module.bias is not None):
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d) and (module.bias is not None):
                nn.init.zeros_(module.bias)

    def forward(self, enc_features: torch.Tensor
                ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Forward function."""
        e4, e8, e16, e32 = enc_features

        e32 = self.enc_conv32(e32)
        e16 = self.enc_conv16(e16)
        e8 = self.enc_conv8(e8)
        e4 = self.enc_conv4(e4)

        # 1/4 scale
        dec = torch.cat([e4, e8, e16, e32], dim=1)  # (b, c, h, w)
        # dec = self.dec_conv(dec)
        dec = dec.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)
        dec = self.dec_linear(dec)
        dec = self.dec_norm(dec)

        outs, attn_weights = self.reducer(dec)

        return outs, attn_weights
