import os
import shutil

IMAGE_PATH = "/data/Datasets/KITTI/2012/raw"
DEPTH_PATH = "/data/Datasets/KITTI/2012/gts"
IMAGE_SAVE_PATH = "extract/KITTI/2012/raw"
DEPTH_SAVE_PATH = "extract/KITTI/2012/gts"

# FILE_LIST_PATH = "train_test_inputs/KITTI/kitti_eigen_train.txt"
FILE_LIST_PATH = "train_test_inputs/KITTI/kitti_eigen_test.txt"

with open(FILE_LIST_PATH, "r", encoding="utf-8") as f:
    for line in f.readlines():
        line_img_file, line_depth_file, _ = line.split(" ")

        img_file = os.path.join(IMAGE_PATH, line_img_file)
        depth_file = os.path.join(DEPTH_PATH, line_depth_file)

        save_img_file = os.path.join(IMAGE_SAVE_PATH, line_img_file)
        save_depth_file = os.path.join(DEPTH_SAVE_PATH, line_depth_file)

        os.makedirs(os.path.dirname(save_img_file), exist_ok=True)
        os.makedirs(os.path.dirname(save_depth_file), exist_ok=True)

        shutil.copy2(img_file, save_img_file)
        shutil.copy2(depth_file, save_depth_file)
