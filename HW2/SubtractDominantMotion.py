import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import affine_transform
from LucasKanadeAffine_old import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
import matplotlib.pyplot as plt


def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)

    ################### TODO Implement Substract Dominent Motion ###################

    M = LucasKanadeAffine(
        image1, image2, threshold, num_iters)
    # M = InverseCompositionAffine(
    # image1, image2, threshold, num_iters)

    # //print("M: ", M)

    # img2_w = affine_transform(np.copy(image2), M)

    errorImg = image2 - np.copy(image1)

    mask[errorImg > tolerance] = 1
    mask[errorImg < tolerance] = 0

    mask = binary_erosion(mask, iterations=1)
    mask = binary_dilation(mask, iterations=5)
    mask = binary_erosion(mask, iterations=1)

    return mask.astype(bool)
