from typing import Dict, Tuple, Optional
import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Normalize


class DepthDataset(Dataset):

    def __init__(self,
                 data_path: str,
                 data_type: str = "NYU",
                 mode: str = "train",
                 img_size: Optional[Tuple[int, int]] = None,
                 height_drop: Tuple[float, int] = (0.0, 0),
                 width_drop: Tuple[float, int] = (0.0, 0),
                 clip_depth: Optional[float] = None,
                 use_right: bool = False,
                 drop_edge: bool = False):
        super().__init__()

        mode = mode.lower()
        if mode not in ("train", "test", "benchmark"):
            raise ValueError(f"DepthDataset mode {mode} is not supported.")
        data_type = data_type.upper()
        if data_type not in ("KITTI", "NYU", "ONLINE"):
            raise ValueError(f"DepthDataset data_type {data_type} is not supported.")

        if (mode == "benchmark") and (data_type != "ONLINE"):
            raise ValueError("Benchmark should only run with ONLINE data type.")

        self.transforms = Compose([
            ImageDepth2Tensor(mode=mode),
            RandomMasking(mode=mode, height_drop=height_drop, width_drop=width_drop, drop_edge=drop_edge)
        ])
        self.data_path = data_path
        self.data_type = data_type
        self.mode = mode
        self.use_right = use_right
        if use_right:
            raise ValueError("DepthDataset currently do not support use_right=True option.")

        # ------------------------ KITTI ------------------------------#
        if self.data_type == "KITTI":
            if mode == "train":
                with open("./dataset/train_test_inputs/KITTI/kitti_eigen_train.txt", 'r') as f:
                    self.filenames = list(f.readlines())
                if img_size is None:
                    self.height, self.width = 352, 704  # depth h, w, randomly crop part of image
                else:
                    self.height, self.width = img_size[0], img_size[1]
                self.do_random_rotate = True
                self.degree = 1.0

            elif mode == "test":
                with open("./dataset/train_test_inputs/KITTI/kitti_eigen_test.txt", 'r') as f:
                    self.filenames = list(f.readlines())
                if img_size is None:
                    self.height, self.width = 376, 1241  # depth h, w, actually unused because of kb_crop (352, 1216)
                else:
                    self.height, self.width = img_size[0], img_size[1]
                self.do_random_rotate = False
                self.degree = None

            self.min_depth = 0.001
            self.max_depth = 80.0
            self.saving_factor = 256
            self.do_kb_crop = True
            self.img_path = os.path.join(self.data_path, "raw")
            self.gt_path = os.path.join(self.data_path, "gts")

        # ------------------------ NYU ------------------------------#
        elif self.data_type == "NYU":
            if mode == "train":
                # with open("./dataset/train_test_inputs/NYU/nyu_train_24k.txt", 'r') as f:
                with open("./dataset/train_test_inputs/NYU/nyu_train_36k.txt", 'r') as f:
                    self.filenames = list(f.readlines())
                if img_size is None:
                    # self.height, self.width = 416, 544  # depth h, w, randomly crop part of image from (427, 565)
                    self.height, self.width = 480, 640  # depth h, w, randomly crop part of image from (480, 640)
                else:
                    self.height, self.width = img_size[0], img_size[1]
                self.do_random_rotate = True
                self.degree = 2.5

            elif mode == "test":
                with open("./dataset/train_test_inputs/NYU/nyu_test.txt", 'r') as f:
                    self.filenames = list(f.readlines())
                if img_size is None:
                    self.height, self.width = 480, 640  # depth h, w, default size of the image.
                else:
                    self.height, self.width = img_size[0], img_size[1]
                self.do_random_rotate = False
                self.degree = None

            self.min_depth = 0.001
            self.max_depth = 10.0
            self.saving_factor = 1000
            self.do_kb_crop = False
            self.img_path = self.data_path
            self.gt_path = self.data_path

        # ------------------------ KITTI ONLINE ------------------------------#
        elif self.data_type == "ONLINE":
            if mode == "train":
                with open("./dataset/train_test_inputs/KITTI/kitti_benchmark_train.txt", 'r') as f:
                    self.filenames = list(f.readlines())
                if img_size is None:
                    self.height, self.width = 352, 704  # depth h, w, randomly crop part of image
                else:
                    self.height, self.width = img_size[0], img_size[1]
                self.do_random_rotate = True
                self.degree = 1.0

                self.img_path = os.path.join(self.data_path, "raw")
                self.gt_path = os.path.join(self.data_path, "gts")

            elif mode == "test":
                with open("./dataset/train_test_inputs/KITTI/kitti_benchmark_val.txt", 'r') as f:
                    self.filenames = list(f.readlines())
                if img_size is None:
                    self.height, self.width = 376, 1241  # depth h, w, actually unused because of kb_crop (352, 1216)
                else:
                    self.height, self.width = img_size[0], img_size[1]
                self.do_random_rotate = False
                self.degree = None

                # self.img_path = os.path.join(self.data_path, "online")
                # self.gt_path = os.path.join(self.data_path, "online")
                self.img_path = self.data_path
                self.gt_path = self.data_path

            elif mode == "benchmark":
                with open("./dataset/train_test_inputs/KITTI/kitti_benchmark_test.txt", 'r') as f:
                    self.filenames = list(f.readlines())
                if img_size is None:
                    self.height, self.width = 376, 1241  # depth h, w, actually unused because of kb_crop (352, 1216)
                else:
                    self.height, self.width = img_size[0], img_size[1]
                self.do_random_rotate = False
                self.degree = None

                # self.img_path = os.path.join(self.data_path, "online")
                # self.gt_path = os.path.join(self.data_path, "online")
                self.img_path = self.data_path
                self.gt_path = None

            self.min_depth = 0.001
            self.max_depth = 88.0
            self.saving_factor = 256
            self.do_kb_crop = True  # TODO this should be FALSE for test set?

        else:  # should not be here
            raise NotImplementedError

        if clip_depth is None:
            clip_depth = self.max_depth
        self.clip_depth = clip_depth  # only applied for training

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict:
        path = self.filenames[idx]
        path = path.replace("\n", "").strip()
        if self.data_type == "KITTI":
            focal = float(path.split()[2])
        else:  # NYU or ONLINE
            focal = 518.8579

        if self.mode != "benchmark":
            image_path_split, depth_path_split = path.split()[:2]
            if image_path_split.startswith("/"):  # NYU train.txt has "/" at the beginning
                image_path_split = image_path_split[1:]
            if depth_path_split.startswith("/"):
                depth_path_split = depth_path_split[1:]

            image_path = os.path.join(self.img_path, image_path_split)
            depth_path = os.path.join(self.gt_path, depth_path_split)

            image = Image.open(image_path)  # (h, w, 3), 8-bit RGB (total 24-bit)
            depth_gt = Image.open(depth_path)  # (h, w), 16-bit grayscale
        else:
            image_path_split = path
            if image_path_split.startswith("/"):  # NYU train.txt has "/" at the beginning
                image_path_split = image_path_split[1:]
            depth_path_split = ""  # dummy

            image_path = os.path.join(self.img_path, image_path_split)
            image = Image.open(image_path)  # (h, w, 3), 8-bit RGB (total 24-bit)
            depth_gt = np.zeros((image.height, image.width), dtype=np.int16)  # dummy
            depth_gt = Image.fromarray(depth_gt)

        if self.do_kb_crop:
            # force KITTI dataset to be the same size (352, 1216)
            # this size of (image, depth) is used fo test.
            assert (depth_gt.height == image.height) and (depth_gt.width == image.width)
            height = image.height
            width = image.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

        # ------------------------ train ------------------------------#
        if self.mode == "train":
            if self.data_type == "NYU":
                # image = image.crop((43, 45, 608, 472))  # (427, 565)
                # depth_gt = depth_gt.crop((43, 45, 608, 472))  # (427, 565)
                depth_gt = np.asarray(depth_gt, dtype=np.float32)
                depth_mask = np.zeros_like(depth_gt)
                depth_mask[45:472, 43:608] = 1
                depth_gt *= depth_mask
                depth_gt = Image.fromarray(depth_gt)

            if self.do_random_rotate:
                random_angle = (random.random() - 0.5) * 2 * self.degree
                image = image.rotate(random_angle, resample=Image.BILINEAR)  # FIX!?
                depth_gt = depth_gt.rotate(random_angle, resample=Image.NEAREST)

        image = np.asarray(image, dtype=np.float32) / 255.0  # (h, w, 3), normalized to [0, 1]
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
        depth_gt = np.expand_dims(depth_gt, axis=2)  # (h, w, 1)

        depth_gt = depth_gt / self.saving_factor  # KITTI 256, NYU 1000
        if self.mode == "train":
            image, depth_gt = self.random_crop(image, depth_gt)
            image, depth_gt = self.train_preprocess(image, depth_gt)

        sample = {'image': image, 'depth': depth_gt, 'focal': focal,
                  'image_path': image_path_split, 'depth_path': depth_path_split}
        sample = self.transforms(sample)  # (h, w, c) -> (c, h, w)
        return sample

    def random_crop(self, img: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = self.height, self.width
        assert (img.shape[0] >= h) and (img.shape[1] >= w) and (img.shape[:2] == depth.shape[:2])
        if (img.shape[0] == h) and (img.shape[1] == w):  # shortcut
            return img, depth

        x = random.randint(0, img.shape[1] - w)
        y = random.randint(0, img.shape[0] - h)
        img = img[y:y + h, x:x + w, :]
        depth = depth[y:y + h, x:x + w, :]
        return img, depth

    def train_preprocess(self, image: np.ndarray, depth_gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Random LR flipping
        if random.random() > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        # if random.random() > 0.5:
        image = self.augment_image(image)
        depth_gt = self.hide_depth(depth_gt)
        return image, depth_gt

    def augment_image(self, image: np.ndarray) -> np.ndarray:
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.data_type == 'NYU':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        image_aug[:, :, 0] *= random.uniform(0.9, 1.1)
        image_aug[:, :, 1] *= random.uniform(0.9, 1.1)
        image_aug[:, :, 2] *= random.uniform(0.9, 1.1)
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def hide_depth(self, depth_gt: np.ndarray) -> np.ndarray:
        depth_gt[depth_gt > self.clip_depth] = 0.0
        return depth_gt


class ImageDepth2Tensor(object):
    def __init__(self, mode: str):
        self.mode = mode
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample: Dict) -> Dict:
        image, depth = sample['image'], sample['depth']

        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)

        sample["image"] = image
        sample["depth"] = depth
        return sample

    def to_tensor(self, pic):
        if not ((isinstance(pic, Image.Image)) or (isinstance(pic, np.ndarray) and (pic.ndim in {2, 3}))):
            raise TypeError('Input pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        else:
            raise NotImplementedError("We believe this should not happen...")


class RandomMasking(object):
    def __init__(self, mode: str,
                 height_drop: Tuple[float, int] = (0.0, 0),
                 width_drop: Tuple[float, int] = (0.2, 4),
                 drop_edge: bool = False):
        """
        :param mode:            "train" or "test"
        :param height_drop:     (maximum chunk ratio, number of chunk drop)
        :param width_drop:      (maximum chunk ratio, number of chunk drop)
        """
        self.mode = mode
        self.height_drop_range = max(min(height_drop[0], 1.0), 0.0)
        self.height_drop_count = max(height_drop[1], 0)
        self.width_drop_range = max(min(width_drop[0], 1.0), 0.0)
        self.width_drop_count = max(width_drop[1], 0)

        self.drop_edge = drop_edge
        if drop_edge:  # force set count to be max 1.
            self.height_drop_count = min(self.height_drop_count, 1)
            self.width_drop_count = min(self.width_drop_count, 1)
            if (self.height_drop_count == 0) and (self.width_drop_count == 0):
                raise ValueError("If drop_edge is ON, you should use at least 1 drop_count.")

    def __call__(self, sample: Dict) -> Dict:
        if self.mode != "train":
            return sample

        image, depth = sample['image'], sample['depth']
        h, w = image.shape[-2:]  # after ImageDepth2Tensor
        mask = torch.ones(h, w, dtype=torch.float32)

        if not self.drop_edge:
            h_drop_max = int((h - 1) * self.height_drop_range)
            w_drop_max = int((w - 1) * self.width_drop_range)

            # height masking
            for i in range(self.height_drop_count):
                h_drop_length = random.randint(0, h_drop_max)
                h_drop_start = random.randint(0, h - h_drop_length)
                h_drop_end = h_drop_start + h_drop_length
                mask[h_drop_start:h_drop_end, :] = 0

            # width masking
            for i in range(self.width_drop_count):
                w_drop_length = random.randint(0, w_drop_max)
                w_drop_start = random.randint(0, w - w_drop_length)
                w_drop_end = w_drop_start + w_drop_length
                mask[:, w_drop_start:w_drop_end] = 0

        else:
            h_keep_max = int((h - 1) * (1.0 - self.height_drop_range))
            w_keep_max = int((w - 1) * (1.0 - self.width_drop_range))

            mask.fill_(0)

            # height masking
            if self.height_drop_count > 0:
                h_keep_length = random.randint(0, h_keep_max)
                h_keep_start = random.randint(0, h - h_keep_length)
                h_keep_end = h_keep_start + h_keep_length
                mask[h_keep_start:h_keep_end, :] = 1

            # width masking
            if self.width_drop_count > 0:
                w_keep_length = random.randint(0, w_keep_max)
                w_keep_start = random.randint(0, w - w_keep_length)
                w_keep_end = w_keep_start + w_keep_length
                mask[:, w_keep_start:w_keep_end] = 1

        # in-place masking
        sample["image"].mul_(mask)
        sample["depth"].mul_(mask)
        return sample
