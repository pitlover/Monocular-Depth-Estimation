from collections import OrderedDict
import torch

ORIG_PATH = "/home/jiyoungkim/.cache/torch/hub/checkpoints/swin_large_patch4_window12_384_22k.pth.bck"
NEW_PATH = "/home/jiyoungkim/.cache/torch/hub/checkpoints/swin_large_patch4_window12_384_22k.pth"

old_params = torch.load(ORIG_PATH, map_location="cpu")
new_params = OrderedDict()

for k, v in old_params["model"].items():
    if "attn_mask" not in k:
        new_params[k] = v

torch.save(new_params, NEW_PATH)
