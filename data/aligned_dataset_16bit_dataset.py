import os
import numpy as np
import torch
from data.base_dataset import BaseDataset
from datasets.make_dataset_aligned import get_file_paths
from util import util
from PIL import Image
import torchvision.transforms.functional as TF

class AlignedDataset16BitDataset(BaseDataset):
    """Dataset for 16-bit grayscale image pairs stored side-by-side (AB format)."""

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.AB_paths = sorted(self.make_dataset(self.dir_AB(opt), opt.max_dataset_size))

    def dir_AB(self, opt):
        phase = 'train' if opt.isTrain else 'test'
        return os.path.join(opt.dataroot, phase)

    def make_dataset(self, dir, max_size=float("inf")):
        paths = get_file_paths(dir)

        # If max_size is float('inf'), just return all paths
        if max_size == float("inf"):
            return paths

        # Convert to int safely
        try:
            max_size = int(max_size)
        except Exception:
            raise ValueError(f"max_size must be an int or float('inf'), got {type(max_size)}: {max_size}")

        return paths[:max_size]

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB_img = Image.open(AB_path)
        AB_np = np.array(AB_img, dtype=np.uint16).astype(np.float32) / 65535.0  # Normalize to [0, 1]

        h, w = AB_np.shape
        w2 = w // 2
        A_np = AB_np[:, :w2]
        B_np = AB_np[:, w2:]

        A = torch.from_numpy(A_np).unsqueeze(0)  # (1, H, W)
        B = torch.from_numpy(B_np).unsqueeze(0)

        if self.opt.load_size != self.opt.crop_size:
            A = TF.resize(A, [self.opt.load_size, self.opt.load_size], antialias=True)
            B = TF.resize(B, [self.opt.load_size, self.opt.load_size], antialias=True)

        A = TF.center_crop(A, [self.opt.crop_size, self.opt.crop_size])
        B = TF.center_crop(B, [self.opt.crop_size, self.opt.crop_size])

        # Normalize to [-1, 1] if needed
        # if not self.opt.no_normalize:
        #     A = A * 2 - 1
        #     B = B * 2 - 1

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)
