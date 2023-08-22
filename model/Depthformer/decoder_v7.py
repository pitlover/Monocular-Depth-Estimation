from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from .layer_utils import ResConvBNBlock, ConvBN, UpscaleConcatAct

from .self_attention import SelfAttentionBlock
from .feed_forward import FeedForwardBlock
from .luna_layer import PreNormLunaLayer
from .vit_layer import ViTLayer


class DepthFormerDecoderV7(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 num_bins: int,
                 num_aux: int,
                 input_channels: Tuple[int, ...],
                 img_size: Tuple[int, int],
                 feedforward_dim: Optional[int] = None,
                 attn_drop_prob: float = 0.1,
                 drop_prob: float = 0.1,
                 act_layer=nn.SiLU):
        super().__init__()

        self.hidden_dim = hidden_dim  # 512
        self.num_heads = num_heads  # 8
        self.input_channels = input_channels  # [24, 40, 64, 176, 2048]
        self.num_inputs = len(input_channels)  # 5
        self.num_bins = num_bins  # 256
        self.num_aux = num_aux  # 256
        num_layers = self.num_inputs  # 5
        self.head_dim = hidden_dim // self.num_heads  # 64

        assert self.num_inputs == 5  # fixed for EfficientNet, 1/2, 1/4, 1/8, 1/16, 1/32
        self.img_size = img_size
        self.embedding_scale = math.sqrt(1 / hidden_dim)

        self.num_aux = (img_size[0] // 32) * (img_size[1] // 32)  # override!
        self.aux_embedding = nn.Parameter(torch.zeros(1, self.num_aux, hidden_dim))
        nn.init.normal_(self.aux_embedding, mean=0, std=self.embedding_scale)

        self.position_embedding = nn.Parameter(torch.zeros(1, hidden_dim, img_size[0] // 32, img_size[1] // 32))
        with torch.no_grad():
            self.position_embedding.data.copy_(
                self.aux_embedding.data.detach().clone().transpose(-1, -2).reshape(self.position_embedding.shape))

        self.internal_dims = [hidden_dim // 8, hidden_dim // 8, hidden_dim // 4, hidden_dim // 2, hidden_dim]
        self.internal_heads = [num_heads // 8, num_heads // 8, num_heads // 4, num_heads // 2, num_heads]

        self.luna_layers = nn.ModuleList([
            PreNormLunaLayer(self.internal_dims[i + 1],
                             hidden_dim,
                             self.internal_dims[i + 1],
                             self.internal_heads[i + 1],
                             feedforward_dim=feedforward_dim,
                             attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer)
            for i in range(num_layers - 1)
        ])
        self.aux_layers = nn.ModuleList([
            ViTLayer(hidden_dim, hidden_dim, num_heads, feedforward_dim=feedforward_dim,
                     attn_drop_prob=attn_drop_prob, drop_prob=drop_prob, act_layer=act_layer)
            for _ in range(num_layers)
        ])
        self.aux_lst_ln = nn.LayerNorm(hidden_dim, eps=1e-5)

        self.encoder_drop = nn.Dropout(drop_prob)  # prevent over-fitting from encoder feature

        self.post_conv_layers = nn.ModuleList([
            ResConvBNBlock(self.input_channels[i] + self.internal_dims[i + 1], self.internal_dims[i], 3,
                           num_layers=2, act_layer=act_layer)
            for i in range(num_layers - 1)
        ])
        self.post_conv_layers.append(
            ResConvBNBlock(self.input_channels[-1], self.internal_dims[-1], 3,
                           num_layers=2, act_layer=act_layer)
        )

        self.upscale_layers = nn.ModuleList([
            UpscaleConcatAct(scale_factor=2, act_layer=act_layer) for _ in range(num_layers - 1)
        ])

        self.bin_regressor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_layer(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_layer(),
            nn.Linear(self.hidden_dim, self.num_bins),
        )
        self.bin_predictor = nn.Sequential(
            ConvBN(self.internal_dims[0], self.internal_dims[0], 3, act_layer=act_layer, use_residual=False),
            nn.Conv2d(self.internal_dims[0], self.num_bins, kernel_size=(1, 1)),
        )

    def forward(self, features: Tuple[torch.Tensor, ...]
                ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        x0, x1, x2, x3, x4 = features
        # x0: (b, ch1, h/2, w/2)  # 24
        # x1: (b, ch2, h/4, w/4)  # 40
        # x2: (b, ch3, h/8, w/8)  # 64
        # x3: (b, ch4, h/16, w/16)  # 176
        # x4: (b, ch5, h/32, w/32)  # 512
        x0 = self.encoder_drop(x0)
        x1 = self.encoder_drop(x1)
        x2 = self.encoder_drop(x2)
        x3 = self.encoder_drop(x3)
        x4 = self.encoder_drop(x4)

        batch_size, _, out_h, out_w = x0.shape

        c4 = self.post_conv_layers[4](x4)  # (b, ch5, h/32, w/32) -> (b, d, h/32, w/32)
        aux = self.aux_embedding.expand(batch_size, self.num_aux, self.hidden_dim)  # (b, K, d)

        pe = self.position_embedding
        c4 = c4 + pe

        # ----------------------------------------------------------------#
        # emb4 = self.position_embeddings[4]
        c4, aux, attn4_1, attn4_2 = self.luna_layers[3](c4, aux)
        aux, _ = self.aux_layers[4](aux)

        c3 = self.upscale_layers[3](x3, c4)
        c3 = self.post_conv_layers[3](c3)  # (b, ch4 + d, h/16, w/16) -> (b, d/2, h/16, w/16)

        # ----------------------------------------------------------------#
        # emb3 = self.position_embeddings[3]
        c3, aux, attn3_1, attn3_2 = self.luna_layers[2](c3, aux)
        aux, _ = self.aux_layers[3](aux)

        c2 = self.upscale_layers[2](x2, c3)
        c2 = self.post_conv_layers[2](c2)  # (b, ch3 + d/2, h/8, w/8) -> (b, d/2, h/8, w/8)

        # ----------------------------------------------------------------#
        # emb2 = self.position_embeddings[2]
        c2, aux, attn2_1, attn2_2 = self.luna_layers[1](c2, aux)
        aux, _ = self.aux_layers[2](aux)

        c1 = self.upscale_layers[1](x1, c2)
        c1 = self.post_conv_layers[1](c1)  # (b, ch2 + d/2, h/4, w/4) -> (b, d/4, h/4, w/4)

        # ----------------------------------------------------------------#
        # emb1 = self.position_embeddings[1]
        c1, aux, attn1_1, attn1_2 = self.luna_layers[0](c1, aux)
        aux, _ = self.aux_layers[1](aux)

        c0 = self.upscale_layers[0](x0, c1)
        c0 = self.post_conv_layers[0](c0)  # (b, ch2 + d/4, h/2, w/2) -> (b, d/4, h/2, w/2)

        # ----------------------------------------------------------------#
        # emb0 = self.position_embeddings[0]
        aux, _ = self.aux_layers[0](aux)
        aux = self.aux_lst_ln(aux)  # (b, n_aux, d)

        # ----------------------------------------------------------------#
        bin_cls = self.bin_predictor(c0)  # (b, n_bins, h/2, w/2) before softmax
        bin_cls = F.softmax(bin_cls, dim=1)

        aux = torch.mean(aux, dim=1)
        # aux = aux[:, 0]
        bin_width = self.bin_regressor(aux)  # (b, n_bins) before relu
        bin_width = F.relu(bin_width) + 0.1
        bin_width = bin_width / torch.sum(bin_width, dim=-1, keepdim=True)  # normalized

        return bin_width, bin_cls, (attn1_1, attn1_2,
                                    attn2_1, attn2_2,
                                    attn3_1, attn3_2,
                                    attn4_1, attn4_2,)


if __name__ == '__main__':
    dummy1 = torch.zeros(4, 24, 240, 320)
    dummy2 = torch.zeros(4, 40, 120, 160)
    dummy3 = torch.zeros(4, 64, 60, 80)
    dummy4 = torch.zeros(4, 176, 30, 40)
    dummy5 = torch.zeros(4, 512, 15, 20)

    dec = DepthFormerDecoderV7(
        hidden_dim=512,
        num_heads=4,
        num_bins=256,
        num_aux=256,
        img_size=(480, 640),
        input_channels=(24, 40, 64, 176, 512),
    )

    bin_width, bin_cls, dummy_attn_weights = dec((dummy1, dummy2, dummy3, dummy4, dummy5))
    print(dec)
    print("Output shape:", bin_width.shape)
    print("Output shape:", bin_cls.shape)
    for weight in dummy_attn_weights:
        print("Attention weight shape:", weight.shape)
