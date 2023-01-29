import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
from matplotlib import animation
from tqdm import tqdm


def updatefig(i, plot_idx=[0, 99, 199, 299, 399]):
    # This function will be called and save images at the above intervals

    rect = lk_res[i]
    pt_topleft = rect[:2]
    pt_bottomright = rect[2:4]
    patch.set_width(pt_bottomright[0] - pt_topleft[0])
    patch.set_height(pt_bottomright[1] - pt_topleft[1])
    patch.set_xy((pt_topleft[0], pt_topleft[1]))
    im.set_array(seq[:, :, i])
    if i in plot_idx:
        plt.savefig("result/tracking_1_3_cars_" + str(i) +
                    ".png", bbox_inches='tight', pad_inches=0)
    return im, patch


parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e2,
                    help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2,
                    help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
seq = np.load("../data/carseq.npy")

# Run LK Tracking
rect = np.array([59, 116, 145, 151]).T

lk_res = []
lk_res.append(rect)
for i in tqdm(range(1, seq.shape[2])):
    pt_topleft = rect[:2]
    pt_bottomright = rect[2:4]
    It = seq[:, :, i-1]
    It1 = seq[:, :, i]
    p = LucasKanade(It, It1, rect, threshold, num_iters)

    with open("lk_log.txt", "a") as log:
        log.write(f"Rect Start: {rect} | ")

    rect = np.concatenate((pt_topleft + p, pt_bottomright + p))

    with open("lk_log.txt", "a") as log:
        log.write(f"Rect End: {rect} ")
        log.write(
            " \n\n+++++++++++++++++++  End Tracking Iteration ++++++++++++++++\n\n")

    lk_res.append(rect)


lk_res = np.array(lk_res)
np.save("carseqrects.npy", lk_res)

# Code for animating, debugging, and saving images.
lk_res = np.load("carseqrects.npy")

fig, ax = plt.subplots(1)
It1 = seq[:, :, 0]
rect = lk_res[0]
pt_topleft = rect[:2]
pt_bottomright = rect[2:4]
patch = patches.Rectangle((pt_topleft[0], pt_topleft[1]), pt_bottomright[0] - pt_topleft[0],
                          pt_bottomright[1] - pt_topleft[1], linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(patch)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
im = ax.imshow(It1, cmap='gray')
ani = animation.FuncAnimation(fig, updatefig, frames=range(lk_res.shape[0]),
                              interval=50, blit=True)
os.makedirs("result", exist_ok=True)
plt.show()

# Sample code for genearting output image grid
fig, axarr = plt.subplots(1, 5)
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plot_idx = [0, 99, 199, 299, 399]
for i in range(5):
    axarr[i].imshow(plt.imread(
        f"result/tracking_1_3_cars_" + str(plot_idx[i]) + ".png"))
    axarr[i].axis('off')
    axarr[i].axis('tight')
    axarr[i].axis('image')
plt.show()
