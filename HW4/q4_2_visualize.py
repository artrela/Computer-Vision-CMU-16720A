import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from q2_1_eightpoint import eightpoint
from q3_2_triangulate import findM2
from q4_1_epipolar_correspondence import epipolarCorrespondence

# Insert your package here


'''
Q4.2: Finding the 3D position of given points based on epipolar correspondence and triangulation
    Input:  temple_pts1, chosen points from im1
            intrinsics, the intrinsics dictionary for calling epipolarCorrespondence
            F, the fundamental matrix
            im1, the first image
            im2, the second image
    Output: P (Nx3) the recovered 3D points
    
    Hints:
    (1) Use epipolarCorrespondence to find the corresponding point for [x1 y1] (find [x2, y2])
    (2) Now you have a set of corresponding points [x1, y1] and [x2, y2], you can compute the M2
        matrix and use triangulate to find the 3D points. 
    (3) Use the function findM2 to find the 3D points P (do not recalculate fundamental matrices)
    (4) As a reference, our solution's best error is around ~2000 on the 3D points. 
'''


def compute3D_pts(temple_pts1, intrinsics, F, im1, im2):

    # ----- TODO -----
    # YOUR CODE HERE
    nPoints = temple_pts1.shape[0]
    temple_pts2 = np.zeros(temple_pts1.shape)

    for i in range(nPoints):
        x1 = temple_pts1[i, 0]
        y1 = temple_pts1[i, 1]

        x2, y2 = epipolarCorrespondence(im1, im2, F, x1, y1)

        temple_pts2[i, 0] = x2
        temple_pts2[i, 1] = y2

    M2, C2, P = findM2(F, temple_pts1, temple_pts2,
                       intrinsics, filename="q3_3.npy")

    M1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])

    C1 = np.matmul(K1, M1)

    if __name__ == "__main__":
        np.savez("results/q4_2.npz", F=F, M1=M1, C1=C1, M2=M2, C2=C2)

    return P


'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
if __name__ == "__main__":

    temple_coords_path = np.load('data/templeCoords.npz')
    # Loading correspondences
    correspondence = np.load('data/some_corresp.npz')
    # Loading the intrinscis of the camera
    intrinsics = np.load('data/intrinsics.npz')
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    # ----- TODO -----
    # YOUR CODE HERE

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    temple_x = temple_coords_path["x1"]
    temple_y = temple_coords_path["y1"]

    temple_pts1 = np.hstack((temple_x, temple_y))

    P = compute3D_pts(temple_pts1, intrinsics, F, im1, im2)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(P[:, 0], P[:, 1], P[:, 2])
    plt.title("3D Point Correspondences")
    plt.show()
