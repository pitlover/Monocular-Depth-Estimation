import os

IMAGE_PATH = "/data/Datasets/KITTI/2012/raw"
DEPTH_PATH = "/data/Datasets/KITTI/2012/gts"

FILE_LIST_PATH = "train_test_inputs/KITTI/kitti_eigen_train.txt"
# FILE_LIST_PATH = "train_test_inputs/KITTI/kitti_eigen_test.txt"

yes_gt = 0
no_gt = 0
wrong_no_img = 0
wrong_no_gt = 0
with open(FILE_LIST_PATH, "r", encoding="utf-8") as f:
    for line in f.readlines():
        line_img_file, line_depth_file = line.replace("\n", "").strip().split(" ")[:2]

        if line_depth_file == "None":
            no_gt += 1
            continue

        img_file = os.path.join(IMAGE_PATH, line_img_file)
        depth_file = os.path.join(DEPTH_PATH, line_depth_file)

        # if not os.path.isfile(depth_file):
        #     depth_file = os.path.join(DEPTH_PATH, "test", line_depth_file)

        if os.path.isfile(img_file) and os.path.isfile(depth_file):
            yes_gt += 1
        else:
            if not os.path.isfile(img_file):
                wrong_no_img += 1
                print(f"Image does not exist: {img_file}")
            if not os.path.isfile(depth_file):
                wrong_no_gt += 1
                print(f"GT does not exist: {depth_file}")

print(f"Valid pairs: {yes_gt}")
print(f"None GT: {no_gt}")
print(f"Wrong Img not exist: {wrong_no_img}")
print(f"Wrong GT not exist: {wrong_no_gt}")
