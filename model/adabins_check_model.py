from collections import OrderedDict
import torch
from model.Adabins.unet_adaptive_bins import UnetAdaptiveBins

CHECKPOINT = "../checkpoint/AdaBins_nyu_rename.pth"
# CHECKPOINT = "../checkpoint/AdaBins_nyu_rename.pth"

model = UnetAdaptiveBins.build(n_bins=256, min_val=0.001, max_val=80.0)

params = torch.load(CHECKPOINT, map_location="cpu")
params = params["model"]
# remove key prefix "module." attached from DDP.
new_params = OrderedDict()
for param_name, param_value in params.items():
    new_param_name = param_name.replace("module.", "")
    new_params[new_param_name] = param_value

model.load_state_dict(new_params, strict=True)

dummy_input = torch.empty(1, 3, 480, 640)
dummy_output, _ = model(dummy_input)

num_params = model.count_params()
print("#Params:", num_params)

encoder_num_params = 0
for v in model.encoder.parameters():
    encoder_num_params += v.numel()
print("#Encoder Params:", encoder_num_params)
print("#New params:", num_params - encoder_num_params)
