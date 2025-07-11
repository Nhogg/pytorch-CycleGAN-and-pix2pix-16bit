"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


import numpy as np
import torch

def tensor2im_16bit(input_image, imtype=np.uint16):
    """
    Converts a torch Tensor into a 16-bit numpy image array suitable for saving.
    Assumes input is in [-1, 1] and converts to [0, 65535].
    """

    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.detach().cpu().float()
    else:
        return input_image

    # Handle shape: [1, C, H, W] or [C, H, W]
    if image_tensor.ndim == 4:
        image_numpy = image_tensor[0].numpy()  # take first in batch
    elif image_tensor.ndim == 3:
        image_numpy = image_tensor.numpy()
    else:
        raise ValueError(f"Unexpected tensor shape: {image_tensor.shape}")

    # Clip to valid range and rescale
    image_numpy = np.clip(image_numpy, -1.0, 1.0)
    image_numpy = ((image_numpy + 1.0) / 2.0) * 65535.0  # [-1,1] → [0,65535]
    image_numpy = np.clip(image_numpy, 0, 65535).astype(imtype)

    # Transpose from [C, H, W] to [H, W, C] if needed
    if image_numpy.ndim == 3:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))

    # Squeeze single-channel last dim (optional, for saving as grayscale)
    if image_numpy.shape[-1] == 1:
        image_numpy = image_numpy[:, :, 0]

    return image_numpy



def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


from PIL import Image
import numpy as np

def save_image(image_numpy, image_path, aspect_ratio=1.0, imtype=np.uint8):
    """
    Save a numpy image to the disk as 8-bit or 16-bit based on dtype.
    """
    if image_numpy.ndim == 3 and image_numpy.shape[0] == 1:
        image_numpy = np.squeeze(image_numpy, axis=0)

    h, w = image_numpy.shape[:2]
    if aspect_ratio > 1.0:
        image_numpy = np.array(Image.fromarray(image_numpy).resize((int(w * aspect_ratio), h)))
    elif aspect_ratio < 1.0:
        image_numpy = np.array(Image.fromarray(image_numpy).resize((w, int(h * aspect_ratio))))

    # Convert numpy to PIL image with correct mode
    if imtype == np.uint16:
        image_pil = Image.fromarray(image_numpy.astype(np.uint16), mode='I;16')
    else:
        image_pil = Image.fromarray(image_numpy.astype(np.uint8))

    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
