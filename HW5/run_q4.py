import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    print(img)
    im1 = skimage.img_as_float(
        skimage.io.imread(os.path.join('../images', img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################

    def sorted_arr(arr):

        # find the xvals
        xvals = []
        for i in arr:
            xvals.append(i[1])
        xvals.sort()
        # use the sorted x vals to sort the original array
        result = []
        for i in xvals:
            for j in arr:
                if i == j[1]:
                    result.append(j)
        return result

    # make the rows by finding the box's whose centroids are within the
    # ref template
    rows = []
    realtime_arr = [bboxes[0]]
    for i in range(1, len(bboxes)):
        curr_bbox = bboxes[i]
        box_center_y = (curr_bbox[2] + curr_bbox[0])/2
        ref_bbox = realtime_arr[-1]
        if box_center_y < ref_bbox[0] or box_center_y > ref_bbox[2]:
            rows.append(realtime_arr)
            realtime_arr = []
            realtime_arr.append(curr_bbox)
        else:
            realtime_arr.append(curr_bbox)
    rows.append(realtime_arr)

    def get_image(bbox):
        # get the box from the image
        y1, x1, y2, x2 = bbox
        ex = bw[y1:y2, x1:x2]
        width = x2 - x1
        height = y2 - y1
        # pad the image based on if its height or width is larger
        if height > width:
            pad_amt = int(height/2)
            ex = np.pad(ex, (pad_amt, pad_amt), "constant")
        elif width > height:
            pad_amt = int(width/2)
            ex = np.pad(ex, (pad_amt, pad_amt), "constant")
        # plt.imshow(ex)
        # plt.show()

        # our image is reversed bw, so swap black and white
        where_0 = np.where(ex == 0)
        where_1 = np.where(ex == 1)
        ex[where_0] = 1
        ex[where_1] = 0
        # plt.imshow(ex)
        # plt.show()

        # erode (since image is bw, we have to erode to have the effect of dialation)
        if img == "04_deep.jpg":
            ex = cv2.erode(ex, np.ones((14, 14)), iterations=1)
        elif img == "02_letters.jpg":
            ex = cv2.erode(ex, np.ones((4, 4)), iterations=1)
        else:
            ex = cv2.erode(ex, np.ones((7, 7)), iterations=1)
        # resize to 32x32 and transpose in one step
        ex = cv2.resize(ex.T, (32, 32))

        # if img == "02_letters.jpg":
        #     plt.imshow(ex)
        #     plt.show()

        # flatten to get proper input size
        ex = ex.flatten()
        return ex

    # for each row, sort the boxes and get the image the bbox represents
    for row in range(len(rows)):
        rows[row] = sorted_arr(rows[row])
        for letter in range(len(rows[row])):
            rows[row][letter] = get_image(rows[row][letter])

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array(
        [_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle', 'rb'))
    ##########################
    ##### your code here #####
    ##########################

    # create a dict that maps probs to characters
    map_char = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
                20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3',
                30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}

    for row in rows:

        # turn row into np array (letters x 1024)
        row = np.vstack(row)

        # do a forward pass in the row
        h1 = forward(row, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)

        # create one long string to add guesses to
        row_letters = ""
        for i in range(probs.shape[0]):
            # for each letter in the row find the guess
            guess = np.argmax(probs[i, :])
            # turn the guess into a character
            row_letters += map_char[guess]
        # print the row
        print(row_letters)
    print("\n")
