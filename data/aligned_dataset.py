import os

import torch
from torchvision import transforms


from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information, preserving
        16 bit resolution.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path)
        AB_np = np.array(AB).astype(np.float32) / 65535.0

        # Split images
        h, w = AB_np.shape
        w2 = w // 2
        A_np = AB_np[:, :w2]
        B_np = AB_np[:, w2:]

        # Convert to torch tensor with shape [1, H, W]
        A_tensor = torch.from_numpy(A_np).unsqueeze(0)
        B_tensor = torch.from_numpy(B_np).unsqueeze(0)

        # Resize and crop manually
        preprocess = transforms.Compose([
            transforms.Resize(self.opt.load_size),
            transforms.CenterCrop(self.opt.crop_size),
        ])

        # Apply transform
        A_pil = transforms.ToPILImage()(A_tensor)
        B_pil = transforms.ToPILImage()(B_tensor)

        A_pil = A_pil.convert('L')
        B_pil = B_pil.convert('L')

        A_tensor = transforms.ToTensor()(preprocess(A_pil))
        B_tensor = transforms.ToTensor()(preprocess(B_pil))

        return {'A': A_tensor, 'B': B_tensor, 'A_paths': AB_path, 'B_paths': AB_path}


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

#C:\Users\natha\Documents\GitHub\pytorch-CycleGAN-and-pix2pix-16bit\datasets\16bit_test
# python3 test.py --dataroot C:\Users\natha\Documents\GitHub\pytorch-CycleGAN-and-pix2pix-16bit\datasets\16bit_test --name mvScatter16b --preprocess scale_width_and_crop --load_size 1536 --crop_size 512 --model pix2pix --num_test 20 --dataset_mode aligned