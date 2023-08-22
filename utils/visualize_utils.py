from typing import List
from PIL import Image
import numpy as np
import matplotlib
import torch
import os
from os.path import join


def colorize(value, vmin=10, vmax=1000, cmap='magma_r'):
    value = value.cpu().numpy()[0, :, :]
    over_mask = value > vmax
    under_mask = value < vmin
    # normalize
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    # print(vmin, vmax, value.min(), value.max())
    value[over_mask] = 255
    value[under_mask] = 255
    img = value.squeeze()
    # img = arr[:, :, :3]

    return img


def visualization(model_output: tuple, data_type: str, min_depth, max_depth, img_path: List[str]):
    data_type = data_type.lower()
    if data_type == "kitti":
        saving_factor = 256
    elif data_type == "nyu":
        saving_factor = 1000
    else:
        raise ValueError(f"No support {data_type} dataset.")

    for i in range(len(img_path)):
        if img_path[i].startswith("/"):
            img_path[i] = img_path[i][1:]
        img_name = img_path[i].split("/")[-1]
        folder = join("output/NYU/GT/train", "/".join(img_path[i].split("/")[:-1]))
        os.makedirs(folder, exist_ok=True)
        pred = model_output[i][0]
        # viz = colorize(pred.unsqueeze(0), vmin=min_depth, vmax=max_depth, cmap='magma_r')
        viz = colorize(pred.unsqueeze(0), vmin=min_depth, vmax=max_depth, cmap='jet')

        Image.fromarray(viz.squeeze()).save(folder + "/" + img_name.replace("jpg", "png"))
