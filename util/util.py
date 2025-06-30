"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def tensor2im_16bit(input_image, imtype=np.uint16):
    """Converts a Tensor into a 16-bit numpy image array."""

    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image

    # Assuming image is 1xCxHxW or CxHxW
    image_numpy = image_tensor[0].cpu().float().numpy() if image_tensor.ndim == 4 else image_tensor.cpu().float().numpy()

    # Clamp to [-1, 1] or [0, 1] depending on your model
    image_numpy = np.clip(image_numpy, -1, 1)

    # Convert from [-1, 1] to [0, 65535]
    image_numpy = ((image_numpy + 1) / 2.0) * 65535.0

    # Convert to uint16
    image_numpy = np.clip(image_numpy, 0, 65535).astype(imtype)

    # If single-channel, shape = (H, W), expand to (H, W, 1)
    if image_numpy.ndim == 2:
        image_numpy = image_numpy[:, :, np.newaxis]

    # Transpose from (C, H, W) to (H, W, C)
    if image_numpy.shape[0] <= 4:  # assume channel first
        image_numpy = np.transpose(image_numpy, (1, 2, 0))

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

from PIL import Image

from PIL import Image
import numpy as np

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a 16-bit numpy image array to disk, handling both grayscale and RGB."""
    if image_numpy.ndim == 3 and image_numpy.shape[2] == 1:
        image_numpy = np.squeeze(image_numpy, axis=2)  # (H, W)
    elif image_numpy.ndim == 3 and image_numpy.shape[2] == 3:
        pass  # RGB, fine as-is
    elif image_numpy.ndim == 2:
        pass  # Already (H, W)
    else:
        raise ValueError(f"Unsupported image shape: {image_numpy.shape}")

    image_pil = Image.fromarray(image_numpy)

    if aspect_ratio != 1.0:
        h, w = image_numpy.shape[0], image_numpy.shape[1]
        new_w = int(w * aspect_ratio)
        new_h = int(h)
        image_pil = image_pil.resize((new_w, new_h), Image.BICUBIC)
    print(image_path, image_numpy.dtype, image_numpy.min(), image_numpy.max())
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
