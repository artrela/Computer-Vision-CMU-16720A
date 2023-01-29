import numpy as np
import cv2
import skimage.io
import skimage.color
from opts import get_opts
from matchPics import matchPics
from planarH import compositeH, orderMatches, computeH_ransac

# Import necessary functions

# Q2.2.4


def warpImage(opts):

    # Open images and opts
    opts = get_opts()
    image1 = cv2.imread('../data/cv_cover.jpg')
    image2 = cv2.imread('../data/cv_desk.png')

    # match features
    matches, locs1, locs2 = matchPics(image1, image2, opts)

    # order features
    x1, x2 = orderMatches(matches, locs1, locs2)

    # compute homography
    H2to1, inliers = computeH_ransac(x1, x2, opts)

    # import the harry potter image
    imageHP = cv2.imread('../data/hp_cover.jpg')
    imageHP = cv2.resize(imageHP, dsize=(image1.shape[1], image1.shape[0]))

    # find composite img
    compositeImg = compositeH(H2to1, imageHP, image2)

    # save the image
    cv2.imwrite("../Results/compositeHP.png", compositeImg)

    # show the image
    # stitch images together and visualize
    cv2.imshow("Source", imageHP)
    cv2.imshow("Destination", image2)
    cv2.imshow("Warped Source", compositeImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)
