import torch
import torch.nn as nn
import timm

from model.ODA2.oda2_swin_transformer import SwinTransformer
from model.ODA2.timm_swin_transformer import swin_large_patch4_window7_224_in22k

# timm_swin = timm.create_model("swin_large_patch4_window7_224_in22k", pretrained=True)
timm_swin = swin_large_patch4_window7_224_in22k(pretrained=True)
print("... TIMM loaded")
oda_swin = SwinTransformer(
    224, patch_size=4, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), window_size=7)
oda_swin.init_weights(pretrained="checkpoint/swin_large_patch4_window7_224_22k.pth")
print("... ODA loaded")

timm_swin: nn.Module
oda_swin: nn.Module
timm_swin.eval()
oda_swin.eval()

dummy_input = torch.empty(2, 3, 224, 224, dtype=torch.float32).normal_()
# del timm_swin.layers[3].blocks[1]
# del oda_swin.layers[3].blocks[1]
assert timm_swin.layers[3].downsample is None
assert oda_swin.layers[3].downsample is None

timm_outs = []
# timm run
print("TIMM run start")
x = timm_swin.patch_embed(dummy_input)
for layer in timm_swin.layers:
    for block in layer.blocks:
        x = block(x)
    timm_outs.append(x)
    if layer.downsample is not None:
        x = layer.downsample(x)
for to in timm_outs:
    print(f"... {to.shape}")
print(f"... TIMM run, {len(timm_outs)}")

print("ODA run start")
oda_outs = oda_swin(dummy_input)
for oo in oda_outs:
    print(f"... {oo.shape}")
print(f"... ODA run, {len(oda_outs)}")

for to, oo in zip(timm_outs, oda_outs):
    print(to.shape, oo.shape)
    difference = torch.abs(to - oo)
    print(difference.sum().item(), difference.mean().item(), difference.max().item())

# for param_name, param in oda_swin.named_parameters():
#     try:
#         timm_param = timm_swin.get_parameter(param_name)
#     except AttributeError:
#         print(f"... param {param_name} not in TIMM")
#         continue
#     param_difference = torch.abs(param - timm_param)
#     print(param_name, param_difference.sum().item(), param_difference.mean().item())

# for param_name, param in timm_swin.named_parameters():
#     try:
#         oda_param = oda_swin.get_parameter(param_name)
#     except AttributeError:
#         print(f"... param {param_name} not in ODA")
#         continue
#     param_difference = torch.abs(param - oda_param)
#     print(param_name, param_difference.sum().item(), param_difference.mean().item())
