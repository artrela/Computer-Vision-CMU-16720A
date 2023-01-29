import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanadeAffine_old import LucasKanadeAffine
from SubtractDominantMotion import SubtractDominantMotion
from tqdm import tqdm
# write your script here, we recommend the above libraries for making your animation
import time


def updatefig(i):
    It = seq[:, :, i-1]
    It1 = seq[:, :, i]
    ax.clear()
    mask = aerial_masks[:, :, i-1]
    im = ax.imshow(It1, cmap="gray")
    ax.imshow(np.ma.masked_where(np.invert(mask), mask), cmap='jet', alpha=0.5)
    if i in plot_idx:
        plt.savefig("result/tracking_3_2_aerial" + str(i) + ".png")
    return im,


parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=500,
                    help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2,
                    help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.19,
                    help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')
plot_idx = [30, 60, 90, 120]

# Running LK + Motion detection
masks = []
time_start = time.time()  # ? end time
for i in tqdm(range(1, seq.shape[2])):  # for i in range(1, 5):
    It = seq[:, :, i-1]
    It1 = seq[:, :, i]

    mask = SubtractDominantMotion(
        It, It1, threshold=threshold, num_iters=num_iters, tolerance=tolerance)
    masks.append(mask)
    # plt.imshow(mask)
    # plt.show()
print(round(((time.time() - time_start) / 60), 2))  # ? end time

masks = np.stack(masks, axis=2)
np.save("aerial_masks.npy", masks)

# Animation
aerial_masks = np.load("aerial_masks.npy")
fig, ax = plt.subplots(1)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

res_masks = []
for i in range(1, seq.shape[2]):
    mask = aerial_masks[:, :, i-1]
    if i in plot_idx:
        res_masks.append(mask)

res_masks = np.stack(res_masks, axis=2)
print(res_masks.shape)
np.save("aerialseqmasks.npy", res_masks)


ani = animation.FuncAnimation(fig, updatefig, frames=range(1, seq.shape[2]),
                              interval=10, blit=True)

plt.show()

# Sample code for genearting output image grid
fig, axarr = plt.subplots(1, len(plot_idx))
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
for i in range(len(plot_idx)):
    axarr[i].imshow(plt.imread(
        f"result/tracking_3_2_aerial" + str(plot_idx[i]) + ".png"))
    axarr[i].axis('off')
    axarr[i].axis('tight')
    axarr[i].axis('image')
plt.show()
