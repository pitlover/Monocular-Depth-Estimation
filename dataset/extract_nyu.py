import os
import shutil

IMAGE_PATH = "/data/Datasets/NYU/data_nyu"
DEPTH_PATH = "/data/Datasets/NYU/data_nyu"
IMAGE_SAVE_PATH = "extract/NYU/data_nyu"
DEPTH_SAVE_PATH = "extract/NYU/data_nyu"

# FILE_LIST_PATH = "train_test_inputs/NYU/nyu_train_24k.txt"
# FILE_LIST_PATH = "train_test_inputs/NYU/nyu_train_36k.txt"
FILE_LIST_PATH = "train_test_inputs/NYU/nyu_test.txt"

with open(FILE_LIST_PATH, "r", encoding="utf-8") as f:
    for line in f.readlines():
        line_img_file, line_depth_file, _ = line.split(" ")

        if line_img_file.startswith("/"):
            line_img_file = line_img_file[1:]
        if line_depth_file.startswith("/"):
            line_depth_file = line_depth_file[1:]

        img_file = os.path.join(IMAGE_PATH, line_img_file)
        depth_file = os.path.join(DEPTH_PATH, line_depth_file)

        save_img_file = os.path.join(IMAGE_SAVE_PATH, line_img_file)
        save_depth_file = os.path.join(DEPTH_SAVE_PATH, line_depth_file)

        os.makedirs(os.path.dirname(save_img_file), exist_ok=True)
        os.makedirs(os.path.dirname(save_depth_file), exist_ok=True)

        shutil.copy2(img_file, save_img_file)
        shutil.copy2(depth_file, save_depth_file)
