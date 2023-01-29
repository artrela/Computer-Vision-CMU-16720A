import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts

import scipy.ndimage
from matplotlib import pyplot as plt
from helper import plotMatches

# Q2.1.6


def rotTest(opts):

    # Read the image and convert to grayscale, if necessary
    img = cv2.imread('../data/cv_cover.jpg')

    # 2x36 array with reach row [rotation, # of matches]
    matchPerRot = np.zeros(shape=(2, 36))

    for i in range(36):

        # Rotate Image (by 10 degrees each pass)
        rotation = i * 10
        img_rotated = scipy.ndimage.rotate(img, rotation)

        # Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(img, img_rotated, opts)

        # display matches at 10 deg, 100 deg, 200 deg(Visualize)
        if i in [1, 10, 20]:
            locs1 = np.fliplr(locs1)
            locs2 = np.fliplr(locs2)
            plotMatches(img, img_rotated, matches, locs1, locs2)

        # Update histogram
        matchPerRot[0, i] = rotation
        matchPerRot[1, i] = matches.shape[0]

    # Display histograms
    plt.bar(matchPerRot[0, :], matchPerRot[1, :])

    plt.xlabel("Rotation (Degrees)")
    plt.ylabel("Number of Matches")
    plt.title("Number of Matches versus Rotation of Image")
    plt.show()


if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
