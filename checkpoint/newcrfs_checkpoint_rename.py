from collections import OrderedDict
import torch

# OLD_CHECKPOINT_PATH = "NewCRFs_kittieigen.ckpt"
# NEW_CHECKPOINT_PATH = "NewCRFs_kitti_rename.pth"
OLD_CHECKPOINT_PATH = "NewCRFs_nyu.ckpt"
NEW_CHECKPOINT_PATH = "NewCRFs_nyu_rename.pth"

old_ckpt = torch.load(OLD_CHECKPOINT_PATH, map_location="cpu")

new_ckpt = OrderedDict()
new_ckpt["model"] = OrderedDict()
for k, v in old_ckpt["model"].items():
    new_k = k.replace("module.", "")
    new_ckpt["model"][new_k] = v

torch.save(new_ckpt, NEW_CHECKPOINT_PATH)
