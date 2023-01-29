from cv2 import VideoCapture
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def create_npy(n_frames=300):  # 300
    cap = cv2.VideoCapture("../data/walking.mov")
    all = []
    i = 0
    while cap.isOpened() and i < n_frames:
        print("Creating Frame: ", i)
        ret, frame = cap.read()
        rgb = np.array(frame)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        grayed_frame = (0.2989 * r + 0.5870 * g + 0.1140 * b)/225
        # filter the high res video for speed
        # grayed_frame = gaussian_filter(grayed_frame, sigma=3)
        all.append(grayed_frame)  # convert to greyscale
        i += 1
    all = np.array(all)
    all = all.transpose(1, 2, 0)
    # person enters view at approx the 10th frame
    return np.array(all[500:-1, 300:-200, 99:299])  # 99:150])


video = create_npy()
# check the location of the template0
plt.imshow(video[:, :, 0])
plt.show()

# find the original location of the rectange

print("Done")

np.save("../data/walking.npy", video)
