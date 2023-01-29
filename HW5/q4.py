import numpy as np
import skimage
import skimage.color
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.restoration
import skimage.segmentation

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# takes a color image
# returns a list of bounding boxes and black_and_white image


def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################

    # denoise the image
    # image = skimage.restoration.denoise_tv_chambolle(
    #     image, weight=0.1, channel_axis=-1)
    # change the image to gray
    image = skimage.color.rgb2gray(image)

    # apply threshold
    thresh = skimage.filters.threshold_otsu(image)
    bw = skimage.morphology.closing(
        image <= thresh, skimage.morphology.square(10)).astype(np.float32)

    # remove artifacts connected to image border
    cleared = skimage.segmentation.clear_border(bw)

    # label image regions
    label_image = skimage.measure.label(cleared)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = skimage.color.label2rgb(
        label_image, image=image, bg_label=0)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image_label_overlay)

    for region in skimage.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 300:
            bboxes.append(region.bbox)

    return bboxes, bw
