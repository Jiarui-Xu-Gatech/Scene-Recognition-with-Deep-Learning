import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################

    all = os.listdir(dir_name)
    count_number = 0
    all_sum = 0
    all_sum_sq = 0
    transform = transforms.Grayscale()
    for folder_name in all:
        #if folder_name == 'train' or folder_name == 'test':
        dir_names = os.path.join(dir_name, folder_name)
        classes = os.listdir(dir_names)
        for class_name in classes:
            images = os.path.join(dir_name, folder_name, class_name)
            images = os.listdir(images)
            for image in images:
                image_file_name = os.path.join(dir_name, folder_name, class_name, image)
                img = transform(Image.open(image_file_name))
                all_pixels = (np.asarray(img).astype('float32')/255).reshape(1, -1)
                #all_pixels = all_pixels / 255
                #all_pixels = all_pixels.reshape(1, -1)
                pixel_sum = np.sum(all_pixels)
                pixel_sum_square = np.sum(np.square(all_pixels))
                count_number += all_pixels.shape[1]
                all_sum += pixel_sum
                all_sum_sq += pixel_sum_square
    # all_sum=all_sum/255
    # all_sum_sq=all_sum_sq/(255*255)
    mean = all_sum / count_number

    std = np.sqrt((1 / (count_number - 1)) * (all_sum_sq - count_number * (mean ** 2)))

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
