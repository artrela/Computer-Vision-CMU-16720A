import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts


def displayMatched(opts, image1, image2):
    """
    Displays matches between two images

    Input
    -----
    opts: Command line args
    image1, image2: Source images
    """

    matches, locs1, locs2 = matchPics(image1, image2, opts)

    # display matched features
    locs1 = np.fliplr(locs1)
    locs2 = np.fliplr(locs2)
    plotMatches(image1, image2, matches, locs1, locs2)


if __name__ == "__main__":

    opts = get_opts()
    image1 = cv2.imread('../data/cv_cover.jpg')
    image2 = cv2.imread('../data/cv_desk.png')

    displayMatched(opts, image1, image2)
