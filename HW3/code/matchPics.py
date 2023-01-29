import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

# Q2.1.4


def matchPics(I1, I2, opts):
    """
    Match features across images

    Input
    -----
    I1, I2: Source images
    opts: Command line args

    Returns
    -------
    matches: List of indices of matched features across I1, I2 [p x 2]
    locs1, locs2: Pixel coordinates of matches [N x 2]
    """

    # print("Computing Matches...\n")

    ratio = opts.ratio  # 'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  # 'threshold for corner detection using FAST feature detector'

    # TODO: Convert Images to GrayScale
    I1g = skimage.color.rgb2gray(I1)
    I2g = skimage.color.rgb2gray(I2)

    # TODO: Detect Features in Both Images
    locs1_temp = corner_detection(I1g, sigma)
    locs2_temp = corner_detection(I2g, sigma)

    # TODO: Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(I1g, locs1_temp)
    desc2, locs2 = computeBrief(I2g, locs2_temp)

    # TODO: Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)

    print(
        f"Matches: {matches.shape[0]} | Locs 1: {locs1.shape[0]} | Locs 2: {locs2.shape[0]}\n")

    # Flip y,x to x, y
    locs1 = np.fliplr(locs1)
    locs2 = np.fliplr(locs2)

    return matches, locs1, locs2
