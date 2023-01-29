from cv2 import destroyAllWindows
import numpy as np
import cv2
import multiprocessing
from matplotlib import animation
import matplotlib.pyplot as plt

from planarH import computeH_ransac, orderMatches, compositeH
from matchPics import matchPics

from opts import get_opts

# Import necessary functions
from helper import loadVid

import time


def preprocessImg(image, shape):

    #!!! preprocess image so that black values don't appear in the image
    zeroInd = np.where(image == 0)
    image[zeroInd] = 1

    image = cv2.resize(image, dsize=shape)

    return image


def arSuperimpose(path: str):
    """ Superimpose a mov file onto the book of the cv book scene."""

    # ? ------------------- Steps ----------------------------

    """
    
    1. Open both videos & the CV cover page
        1.a Remove the black bars from the video
            - Find where the black bars are on the first frame
            - Crop the first frame
            - Make all the rest of the frames the same size
        1.b Each frame of the video to be warped to be the same size as the book
            - Take the pixels from the cv book and only take those from the video
    2. Find which one is shorter: make the final video that many frames
        2.a find the num frames for each vid
        2.b Which ever video is shorter, clip the longer frame to match
        2.c Change black pixels from warp video to slighly less black
    3. Match the cv_book to each frame of the cv_video
        - Input: I1, I2, opts
        - Output: matches, locs1, locs2
    4. Order the matches from 3
        - Input: matches, locs1, locs2
        - Output: x1, x2
    5. Compute the homography of each frame with x1, x2
        - Input: Locs1, Locs2, opts
        - Output: H2to1
    6. Using that homography calculated in 3, apply it to each kunfu panda frame
    7. Save the video 
    
    """

    opts = get_opts()
    n_cpu = multiprocessing.cpu_count()

    # 1.
    print("Openiing the videos...")
    warp_vid = loadVid(path)
    cv_video = loadVid("../data/book.mov")
    cv_book = cv2.imread("../data/cv_cover.jpg")
    print("Done opening the videos!")

    # 1.a
    print("Resizing the video to be warped...")
    frame0 = warp_vid[0, :, :, :]
    y_nonzero, x_nonzero, _ = np.nonzero(frame0 > 20)
    warp_vid = warp_vid[:, np.min(y_nonzero):np.max(y_nonzero),
                        np.min(x_nonzero):np.max(x_nonzero), :]

    # 1.b
    warpCenterY = int(warp_vid.shape[0] / 2)

    aspectRatio = cv_book.shape[0] / cv_book.shape[1]
    numCols = int(warp_vid.shape[1] / aspectRatio)
    cropLeft = warpCenterY
    cropRight = warpCenterY + (numCols)

    # 1.c
    warp_vid = warp_vid[:, :, cropLeft:cropRight, :]
    print("Completed resizing!")

    # 2.
    # 2.a
    warp_frames = warp_vid.shape[0]
    cv_frames = cv_video.shape[0]

    # 2.b
    totalFrames = 0
    if warp_frames > cv_frames:
        warp_vid = warp_vid[0:cv_frames, :, :, :]
        totalFrames = cv_frames
    else:
        cv_video = cv_video[0:warp_frames, :, :, :]
        totalFrames = warp_frames

    # # 2.c
    print("Preprocessing the Ar video...")
    frames = [(warp_vid[i, :, :, :], (cv_book.shape[1], cv_book.shape[0]))
              for i in range(totalFrames)]
    with multiprocessing.Pool(processes=n_cpu) as pool:
        results = pool.starmap(preprocessImg, frames)
    pool.close()
    warp_vid = np.stack(results, axis=0)

    print("Done preprocessing the video!")

    # 3.
    print("Find the match points between all frames...")
    matchParams = [(cv_book, cv_video[i, :, :, :], opts)
                   for i in range(totalFrames)]
    with multiprocessing.Pool(processes=n_cpu) as pool:
        results = pool.starmap(matchPics, matchParams)
    pool.close()
    print("Done finding matches!")

    # 4.
    print("Order the matches between pictures...")
    orderParams = results
    with multiprocessing.Pool(processes=n_cpu) as pool:
        results = pool.starmap(orderMatches, orderParams)
    pool.close()
    print("Done ordering matches!")

    # 5.
    print("Find the Homography between image 2 and 1...")
    orderResults = results
    hParams = [(match[0], match[1], opts) for match in orderResults]
    with multiprocessing.Pool(processes=n_cpu) as pool:
        results = pool.starmap(computeH_ransac, hParams)
    pool.close()
    print("Done finding homographies!")

    # 6.
    print("Superimposing video onto CV Book scene...")
    H2to1_all = results
    finalParams = [(H2to1_all[i][0], warp_vid[i, :, :, :], cv_video[i, :, :, :])
                   for i in range(totalFrames)]

    with multiprocessing.Pool(processes=n_cpu) as pool:
        results = pool.starmap(compositeH, finalParams)
    pool.close()
    print("Done generating video... saving result...")

    # 7.
    finalVideo = np.stack(results, axis=0)

    out = cv2.VideoWriter('../data/outpy.avi', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 24, (finalVideo.shape[2], finalVideo.shape[1]))

    for i in range(totalFrames):
        out.write(finalVideo[i, :, :, :])
    out.release()


if __name__ == "__main__":
    arSuperimpose("../data/ar_source.mov")
