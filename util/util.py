"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def tensor2im_16bit(input_image):
    """"Converts a Tensor array into a 16 bit numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    image_tensor = input_image.detach().cpu()[0]
    image_numpy = image_tensor.numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    image_numpy = (image_numpy * 65535.0).astype(np.uint16)
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

def save_image(image_numpy, image_path, aspect_ratio=1.0) -> None:
    """Save a numpy image to disk, preserving 16-bit depth."""
    # Ensure the array is 2D or 3D with shape (H, W)
    if image_numpy.ndim == 3 and image_numpy.shape[0] == 1:
        image_numpy = image_numpy[0]  # Remove channel dimension

    # Create the PIL image (for 16-bit grayscale)
    image_pil = Image.fromarray(image_numpy, mode='I;16')

    # Apply aspect ratio adjustment
    w, h = image_pil.size
    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((w, int(h * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(w / aspect_ratio), h), Image.BICUBIC)

    # Save
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
