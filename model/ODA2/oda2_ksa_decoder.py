from typing import Union, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
import torch.utils.checkpoint as checkpoint

from .oda2_layer_utils import PyramidPoolingModule, ConvBN
from .oda2_swin_transformer import (WindowAttention, SwinMLP, SwinWindowing, SwinTransformerBlock,
                                    DropPath, _SWIN_PADDING_MODE)


class ConvMLP(nn.Module):
    """ Multilayer perceptron with intermediate ConvDW"""

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 kernel_size: int = 5,
                 drop_prob: float = 0.0,
                 act_layer=nn.GELU) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        # depth-wise conv
        self.conv = ConvBN(hidden_features, hidden_features, kernel_size=kernel_size,
                           conv_groups=hidden_features, act_layer=nn.GELU)

        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)

        b, h, w, c = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()

        x = self.fc2(x)
        x = self.drop(x)
        return x


class KernelWindowAttention(nn.Module):

    def __init__(self,
                 dim: int,
                 enc_dim: int,
                 window_size: Union[int, Tuple[int, int]],
                 num_heads: int,
                 qkv_bias: bool = True,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
        self.num_heads = num_heads
        if dim % num_heads != 0:
            raise ValueError(f"Dim {dim} is not divisible by num_heads {num_heads}.")

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(enc_dim, enc_dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop_prob)
        self.proj_drop = nn.Dropout(drop_prob)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,
                x: torch.Tensor,
                enc: torch.Tensor) -> torch.Tensor:
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            enc: encoder features with shape of (num_windows*B, N, C')
            # TODO maybe we need specialized masking
        """
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)  # (b, nh, n, hd)
        _, _, enc_c = enc.shape
        kv = self.kv(enc).reshape(b, n, 2, self.num_heads, enc_c // self.num_heads).permute(2, 0, 3, 4, 1)
        k, v = kv[0], kv[1]  # (b, nh, enc_hd, n)

        attn_scale = math.sqrt(1 / n)
        attn = torch.matmul(k, q).transpose(-2, -1)  # (b, nh, enc_hd, hd) -> (b, nh, hd, enc_hd)
        attn *= attn_scale
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v)  # (b, nh, hd, enc_hd) x (b, nh, enc_hd, n) = (b, nh, hd, n)
        x = x.permute(0, 3, 1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class KSATransformerBlock(nn.Module):
    """ KSA Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_prob (float, optional): Dropout rate. Default: 0.0
        attn_drop_prob (float, optional): Attention dropout rate. Default: 0.0
        path_drop_prob (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
    """

    def __init__(self,
                 dim: int,
                 enc_dim: int,
                 num_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.0,
                 path_drop_prob: float = 0.0,
                 act_layer=nn.GELU) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if not (0 <= self.shift_size < self.window_size):
            raise ValueError(f"shift_size {shift_size} must in [0, window_size {window_size})")
        mlp_hidden_dim = int(dim * mlp_ratio)

        # TODO is order (1) kernel (2) self (3) ff optimal?
        self.norm_kernel = nn.LayerNorm(dim)
        self.norm_enc = nn.LayerNorm(enc_dim)
        self.kernel_attn = KernelWindowAttention(
            dim, enc_dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop_prob=attn_drop_prob, drop_prob=drop_prob)

        self.norm_ff1 = nn.LayerNorm(dim)
        self.mlp1 = SwinMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_prob=drop_prob)
        # self.mlp1 = ConvMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_prob=drop_prob)

        self.norm_attn = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop_prob=attn_drop_prob, drop_prob=drop_prob)

        self.norm_ff2 = nn.LayerNorm(dim)
        self.mlp2 = SwinMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_prob=drop_prob)
        # self.mlp2 = ConvMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_prob=drop_prob)

        self.windowing = SwinWindowing(window_size=window_size)
        self.drop_path = DropPath(path_drop_prob) if (path_drop_prob > 0) else nn.Identity()
        self.H: Optional[int] = None  # placeholder
        self.W: Optional[int] = None  # placeholder

    def forward(self,
                x: torch.Tensor,
                enc: torch.Tensor,
                mask_matrix: torch.Tensor) -> torch.Tensor:
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            enc: Input feature, tensor size (B, H*W, C').
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        b, n, c = x.shape
        h, w = self.H, self.W
        if n != h * w:
            raise ValueError(f"Input shape {tuple(x.shape)} does not match with size ({h}, {w}).")

        _, _, enc_c = enc.shape
        x = x.view(b, h, w, c)
        enc = enc.view(b, h, w, enc_c)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b), mode=_SWIN_PADDING_MODE)
        enc = F.pad(enc, (0, 0, pad_l, pad_r, pad_t, pad_b), mode=_SWIN_PADDING_MODE)
        _, hp, wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_enc = torch.roll(enc, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            shifted_enc = enc
            attn_mask = None

        # partition windows
        enc_windows = self.windowing.window_partition(shifted_enc)  # (b x num_windows, window_size, window_size, enc_c)
        enc_windows = enc_windows.view(-1, self.window_size * self.window_size, enc_c)  # (~, window_size^2, c')

        x_windows = self.windowing.window_partition(shifted_x)  # (b x num_windows, window_size, window_size, c)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # (~, window_size^2, c)

        # K-MSA
        shortcut = x_windows
        x_windows = self.norm_kernel(x_windows)
        enc_windows = self.norm_enc(enc_windows)
        k_windows = self.kernel_attn(x_windows, enc_windows)
        k_windows = shortcut + self.drop_path(k_windows)

        # reverse cyclic shift
        if self.shift_size > 0:
            k_windows = torch.roll(k_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        # merge windows
        k_windows = k_windows.view(-1, self.window_size, self.window_size, c)
        k_windows = self.windowing.window_reverse(k_windows)  # (b, hp, wp, c)

        # FFN1
        shortcut = k_windows
        k_windows = self.norm_ff1(k_windows)
        k_windows = self.mlp1(k_windows)
        k_windows = shortcut + self.drop_path(k_windows)

        # cyclic shift
        if self.shift_size > 0:
            k_windows = torch.roll(k_windows, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        k_windows = self.windowing.window_partition(k_windows)  # (b x num_windows, window_size, window_size, c)
        k_windows = k_windows.view(-1, self.window_size * self.window_size, c)  # (~, window_size^2, c)

        # W-MSA/SW-MSA
        shortcut = k_windows
        k_windows = self.norm_attn(k_windows)
        attn_windows = self.attn(k_windows, mask=attn_mask)  # (~, window_size^2, c)
        attn_windows = shortcut + self.drop_path(attn_windows)

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_windows = torch.roll(attn_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        attn_windows = self.windowing.window_reverse(attn_windows)  # (b, hp, wp, c)

        # FFN2
        shortcut = attn_windows
        attn_windows = self.norm_ff2(attn_windows)
        attn_windows = self.mlp2(attn_windows)
        attn_windows = shortcut + self.drop_path(attn_windows)

        # finalize
        x = attn_windows
        if (pad_r > 0) or (pad_b > 0):
            x = x[:, :h, :w, :].contiguous()
        x = x.view(b, h * w, c)

        return x


class KSATransformerStage(nn.Module):
    """ A basic KSA Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_prob (float, optional): Dropout rate. Default: 0.0
        attn_drop_prob (float, optional): Attention dropout rate. Default: 0.0
        path_drop_prob (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim: int,
                 enc_dim: int,
                 depth: int,
                 num_heads: int,
                 window_size: int = 7,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_prob: float = 0.0,
                 attn_drop_prob: float = 0.0,
                 path_drop_prob: Union[float, Tuple[float, ...]] = 0.0,
                 use_ksa: bool = True,
                 upsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.use_ksa = use_ksa
        if use_ksa:  # 1/16, 1/8, 1/4
            self.blocks = nn.ModuleList([
                KSATransformerBlock(
                    dim=dim,
                    enc_dim=enc_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_prob=drop_prob,
                    attn_drop_prob=attn_drop_prob,
                    path_drop_prob=path_drop_prob[i] if isinstance(path_drop_prob, (tuple, list)) else path_drop_prob,
                ) for i in range(depth)
            ])
        else:  # 1/32
            self.blocks = nn.ModuleList([
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_prob=drop_prob,
                    attn_drop_prob=attn_drop_prob,
                    path_drop_prob=path_drop_prob[i] if isinstance(path_drop_prob, (tuple, list)) else path_drop_prob,
                ) for i in range(depth)
            ])

        self.windowing = SwinWindowing(window_size=window_size)

        # patch un-merging layer
        if upsample is not None:
            self.upsample = upsample(dim=dim)
        else:
            self.upsample = None

    def forward(self,
                x: torch.Tensor,
                enc: torch.Tensor,
                h: int, w: int) -> Tuple[torch.Tensor, int, int, torch.Tensor, int, int]:
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            enc: Input feature, tensor size (B, H*W, C').
            h: Spatial height of the input feature.
            w: Spatial width of the input feature.
        """

        # calculate attention mask for SW-MSA
        hp = int(np.ceil(h / self.window_size)) * self.window_size
        wp = int(np.ceil(w / self.window_size)) * self.window_size
        img_mask = torch.zeros(1, hp, wp, 1, dtype=x.dtype, device=x.device)  # (1, hp, wp, 1)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        cnt = 0
        for _h in h_slices:
            for _w in w_slices:
                img_mask[:, _h, _w, :] = cnt
                cnt += 1

        mask_windows = self.windowing.window_partition(img_mask)  # (num_windows, window_size, window_size, 1)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # if min(h, w) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     for blk in self.blocks:
        #         blk.shift_size = 0

        if self.use_ksa:
            for blk in self.blocks:
                blk: KSATransformerBlock
                blk.H, blk.W = h, w
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x, enc, attn_mask)
                else:
                    x = blk(x, enc, attn_mask)
        else:
            for blk in self.blocks:
                blk: SwinTransformerBlock
                blk.H, blk.W = h, w
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x, attn_mask)
                else:
                    x = blk(x, attn_mask)

        if self.upsample is not None:
            x_up = self.upsample(x, h, w)
            h_up, w_up = h * 2, w * 2
            return x, h, w, x_up, h_up, w_up
        else:
            return x, h, w, x, h, w


class PatchUnMerging(nn.Module):
    """ Patch UnMerging Layer

    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # self.norm = nn.LayerNorm(dim // 4)
        # self.expansion = nn.Linear(dim // 4, dim // 2, bias=False)
        self.expansion = ConvBN(dim // 4, dim // 2, 3, act_layer=nn.GELU)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            h: Spatial height of the input feature.
            w: Spatial width of the input feature.
        """
        b, hw, d = x.shape
        assert hw == h * w

        x = x.view(b, h, w, 4, d // 4)
        x0 = x[:, :, :, 0]
        x1 = x[:, :, :, 1]
        x2 = x[:, :, :, 2]
        x3 = x[:, :, :, 3]

        y = torch.zeros(b, 2 * h, 2 * w, d // 4, device=x.device, dtype=x.dtype)
        y[:, 0::2, 0::2, :] = x0
        y[:, 1::2, 0::2, :] = x1
        y[:, 0::2, 1::2, :] = x2
        y[:, 1::2, 1::2, :] = x3

        # y = self.norm(y)
        y = y.permute(0, 3, 1, 2)
        y = self.expansion(y)
        y = y.permute(0, 2, 3, 1)
        y = y.view(b, 4 * h * w, d // 2)

        return y


class KSATransformerRegDecoder(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_prob (float): Dropout rate.
        attn_drop_prob (float): Attention dropout rate. Default: 0.
        path_drop_prob (float): Stochastic depth rate. Default: 0.2.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dec_dim: int = 1024,
                 enc_dims: Tuple[int, int, int, int] = (192, 384, 768, 1536),
                 depths: Tuple[int, ...] = (2, 2, 2, 2),
                 num_heads: Tuple[int, ...] = (4, 8, 16, 32),
                 window_size: int = 7,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_prob: float = 0.0,
                 attn_drop_prob: float = 0.0,
                 path_drop_prob: float = 0.2,
                 use_checkpoint: bool = False):
        super().__init__()

        self.num_layers = len(depths)
        self.dec_dim = dec_dim
        self.enc_dims = enc_dims
        assert len(enc_dims) == 4

        num_features = [int(dec_dim / (2 ** (self.num_layers - i - 1))) for i in range(self.num_layers)]
        self.num_features = tuple(num_features)

        self.ppm32 = PyramidPoolingModule(enc_dims[3], proj_ch=512, out_ch=dec_dim,
                                          spatial_sizes=(1, 2, 3, 6), act_layer=nn.GELU)
        # self.enc_conv16 = ConvBN(enc_dims[2], enc_dims[2], 3, act_layer=nn.GELU)
        # self.enc_conv8 = ConvBN(enc_dims[1], enc_dims[1], 3, act_layer=nn.GELU)
        # self.enc_conv4 = ConvBN(enc_dims[0], enc_dims[0], 3, act_layer=nn.GELU)
        self.enc_conv16 = ConvBN(enc_dims[2], num_features[2], 3, act_layer=nn.GELU)
        self.enc_conv8 = ConvBN(enc_dims[1], num_features[1], 3, act_layer=nn.GELU)
        self.enc_conv4 = ConvBN(enc_dims[0], num_features[0], 3, act_layer=nn.GELU)

        # stochastic depth
        pdp = [x.item() for x in torch.linspace(0, path_drop_prob, sum(depths))]  # stochastic depth decay rule

        # build stages
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = KSATransformerStage(
                dim=num_features[i_layer],
                # enc_dim=enc_dims[i_layer],
                enc_dim=num_features[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_prob=drop_prob,
                attn_drop_prob=attn_drop_prob,
                path_drop_prob=tuple(pdp[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]),
                use_ksa=(i_layer < self.num_layers - 1),
                # use_ksa=(i_layer > 0),
                upsample=PatchUnMerging if (i_layer > 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # self.norm4 = nn.LayerNorm(self.num_features[0])
        # self.norm8 = nn.LayerNorm(self.num_features[1])
        # self.norm16 = nn.LayerNorm(self.num_features[2])

        out_ch = min(self.num_features[0], 128)
        self.dec_conv4 = ConvBN(self.num_features[0], out_ch, 3, act_layer=nn.GELU)
        # self.dec_conv8 = ConvBN(self.num_features[1], 128, 3, act_layer=nn.GELU)
        # self.dec_conv16 = ConvBN(self.num_features[2], 128, 3, act_layer=nn.GELU)

        # Note: intentionally we use padding 0
        # self.out_conv = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
        #     nn.GELU(),
        #     nn.Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        # )
        self.out_conv = nn.Conv2d(out_ch, 1, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.out_sigmoid = nn.Sigmoid()

    def forward(self, enc_features: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        e4, e8, e16, e32 = enc_features

        # nhwc -> nchw
        # e32 = e32.permute(0, 3, 1, 2).contiguous()
        # e16 = e16.permute(0, 3, 1, 2).contiguous()
        # e8 = e8.permute(0, 3, 1, 2).contiguous()
        # e4 = e4.permute(0, 3, 1, 2).contiguous()

        e32 = self.ppm32(e32)
        e16 = self.enc_conv16(e16)
        e8 = self.enc_conv8(e8)
        e4 = self.enc_conv4(e4)

        # nchw -> nhwc
        e32 = e32.permute(0, 2, 3, 1).contiguous()
        e16 = e16.permute(0, 2, 3, 1).contiguous()
        e8 = e8.permute(0, 2, 3, 1).contiguous()
        e4 = e4.permute(0, 2, 3, 1).contiguous()

        # --------------------------------
        # 1/32 scale
        batch_size, h32, w32, _ = e32.shape
        e32 = e32.view(batch_size, h32 * w32, -1)
        _, _, _, d16, _, _ = self.layers[3](e32, e32, h32, w32)

        # --------------------------------
        # 1/16 scale
        _, h16, w16, _ = e16.shape
        e16 = e16.view(batch_size, h16 * w16, -1)
        out16, _, _, d8, _, _ = self.layers[2](d16, e16, h16, w16)
        # out16 = self.norm16(out16)
        # out16 = out16.transpose(-1, -2).reshape(batch_size, -1, h16, w16)
        # out16 = self.dec_conv16(out16)
        # out16 = F.interpolate(out16, scale_factor=4, mode="bilinear", align_corners=True)

        # --------------------------------
        # 1/8 scale
        _, h8, w8, _ = e8.shape
        e8 = e8.view(batch_size, h8 * w8, -1)
        out8, _, _, d4, _, _ = self.layers[1](d8, e8, h8, w8)
        # out8 = self.norm8(out8)
        # out8 = out8.transpose(-1, -2).reshape(batch_size, -1, h8, w8)
        # out8 = self.dec_conv8(out8)
        # out8 = F.interpolate(out8, scale_factor=2, mode="bilinear", align_corners=True)

        # --------------------------------
        # 1/4 scale
        _, h4, w4, _ = e4.shape
        e4 = e4.view(batch_size, h4 * w4, -1)
        out4, _, _, _, _, _ = self.layers[0](d4, e4, h4, w4)
        # out4 = self.norm4(out4)
        out4 = out4.transpose(-1, -2).reshape(batch_size, -1, h4, w4)
        out4 = self.dec_conv4(out4)

        # --------------------------------
        # aggregate 1/4, 1/8, 1/16
        # out = out4 + out8 + out16
        out = out4
        out = self.out_conv(out)
        out = self.out_sigmoid(out)

        return out
