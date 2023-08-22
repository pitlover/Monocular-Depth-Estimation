from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
# import timm

from .oda_swin_transformer import SwinTransformer


class ODASwinEncoder(nn.Module):

    def __init__(self,
                 input_size: Tuple[int, int],  # NOT size of x, but desired input size to the model.
                 model_name: str = "swin_large_patch4_window12_384_in22k",
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1,
                 path_drop_prob: float = 0.1,
                 ) -> None:
        super().__init__()

        """TIMM SwinTransformers
        
        * 224x224 pixels -> -> 56x56 patches -> 8x8 windows (128 ch, patch resolution = 4x4 pixels) 
            -> (down-sampling: 28x28 patches) -> 4x4 windows (128 ch -> 512 ch -> 256 ch, patch res. = 8x8 pixels)
            -> (down-sampling: 14x14 patches) -> 2x2 windows (256 ch -> 1024 ch -> 512 ch, patch res. = 16x16 pixels) 
            -> (down-sampling: 7x7 patches) -> 1x1 windows  (512 ch -> 2048 ch -> 1024 ch, patch res = 32x32 pixels)
        
        'swin_base_patch4_window7_224': 
        'swin_base_patch4_window7_224_in22k'
        'swin_base_patch4_window12_384'
        'swin_base_patch4_window12_384_in22k'
        'swin_large_patch4_window7_224'
        'swin_large_patch4_window7_224_in22k'
        'swin_large_patch4_window12_384'
        'swin_large_patch4_window12_384_in22k' ******** OURS ********
        'swin_small_patch4_window7_224'
        'swin_tiny_patch4_window7_224'
        
        For KITTI, (352, 1216) is inference size.
        - 352 = 32x11, 1216 = 32x38
        - we will instead use (384, 1152) as inference size by bilinear resize and window size of 12.
        - 384 = 32x12, 1152 = 32x36
        - in other words, Encoder will take (384, 1152) as input size.
            - (stage 1): 96x288 patches -> 8x24 windows (192 ch, patch res. = 4x4), output shape = (96x288, 192)
            - (stage 2): 48x144 patches -> 4x12 windows (384 ch, patch res. = 8x8), output shape = (48x144, 384)
            - (stage 3): 24x72 patches -> 2x6 windows (768 ch, patch res. = 16x16), output shape = (24x72, 768)
            - (stage 4): 12x36 patches -> 1x3 windows (1536 ch, patch res. = 32x32), output shape = (12x36, 1536)
                   
        For NYUv2, (480, 640) is inference size.
        - TODO
        
        """
        self.window_size = 12
        if (input_size[0] % (self.window_size * 32) != 0) or (input_size[1] % (self.window_size * 32) != 0):
            raise ValueError(f"Input size {input_size} should be multiple of window size {self.window_size} x32.")

        self.input_size = input_size
        # self.backbone = timm.create_model(model_name, pretrained=True, img_size=input_size, pretrained_strict=False)
        self.backbone = SwinTransformer(
            img_size=384, num_classes=21841, patch_size=4, window_size=12,
            embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
            drop_rate=drop_prob, attn_drop_rate=attn_drop_prob, drop_path_rate=path_drop_prob,
        )
        pretrained_params = torch.load(
            "/home/jiyoungkim/.cache/torch/hub/checkpoints/swin_large_patch4_window12_384_22k.pth",
            map_location="cpu"
        )
        self.backbone.load_state_dict(pretrained_params, strict=True)
        del self.backbone.norm
        del self.backbone.avgpool
        del self.backbone.head

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        orig_h, orig_w = x.shape[-2:]
        if (orig_h, orig_w) != self.input_size:
            # (352, 1216) -> (384, 1152)
            # (352, 704) -> (384, 768)
            new_h = int(np.round(orig_h / 384) * 384)
            new_w = int(np.round(orig_w / 384) * 384)

            if (new_h, new_w) != self.input_size:
                self.input_size = (new_h, new_w)
                self.backbone.reset_resolution(self.input_size)

            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=True)

        x = self.backbone.patch_embed(x)
        x = self.backbone.pos_drop(x)

        stages = []
        for layer in self.backbone.layers:
            for block in layer.blocks:
                x = block(x)
            stages.append(x)
            if layer.downsample is not None:
                x = layer.downsample(x)

        return stages


if __name__ == '__main__':
    net = ODASwinEncoder(input_size=(384, 1152))
    dummy_input = torch.empty(1, 3, 352, 1216)
    dummy_outputs = net(dummy_input)
    for do in dummy_outputs:
        print(do.shape)
