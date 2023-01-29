import numpy as np
import matplotlib.pyplot as plt

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix
import math

# Insert your package here


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
'''


def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation

    c1_p1 = C1[0, :]
    c1_p2 = C1[1, :]
    c1_p3 = C1[2, :]

    c2_p1 = C2[0, :]
    c2_p2 = C2[1, :]
    c2_p3 = C2[2, :]

    N = pts1.shape[0]
    X = np.zeros(shape=(N, 3))
    total_error = 0

    for i in range(N):

        x1, y1 = pts1[i, 0], pts1[i, 1]

        x2, y2 = pts2[i, 0], pts2[i, 1]

        row1 = y1 * c1_p3 - c1_p2
        row2 = c1_p1 - x1 * c1_p3
        row3 = y2 * c2_p3 - c2_p2
        row4 = c2_p1 - x2 * c2_p3

        # (1)
        A = np.vstack((row1, row2, row3, row4))
        # (2)
        U, S, vT = np.linalg.svd(A)

        v = vT.T

        result3d = v[:, -1]

        result3d_homog = result3d / result3d[-1]

        image1_2d_pts = np.matmul(C1, result3d_homog.T)
        image2_2d_pts = np.matmul(C2, result3d_homog.T)

        image1_2d_pts /= image1_2d_pts[-1]
        image2_2d_pts /= image2_2d_pts[-1]

        image1_xy = image1_2d_pts[0:2]
        image2_xy = image2_2d_pts[0:2]

        # (3)
        image1_error = np.linalg.norm(image1_xy - pts1[i, :]) ** 2
        image2_error = np.linalg.norm(image2_xy - pts2[i, :]) ** 2

        # (4)
        X[i, :] = result3d_homog[0:3]
        total_error += (image1_error + image2_error)

    return X, total_error


'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''


def findM2(F, pts1, pts2, intrinsics, filename='q3_3.npz'):
    '''
    Q2.2: Function to find the camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)

    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track 
        of the projection error through best_error and retain the best one. 
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'. 

    '''

    K1 = intrinsics["K1"]
    K2 = intrinsics["K2"]

    # use essential matrix to retrieve E, so you can get M2 matrix
    E = essentialMatrix(F, K1, K2)

    M2s = camera2(E)

    # init camera matrix for triangulating
    M1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])

    C1 = np.matmul(K1, M1)

    # init loop vars
    lowest_error = math.inf
    best_M2 = 0
    possibilities = 4
    # Loop for all 4 possibilities to find the best projective matrix
    for i in range(possibilities):
        M2_possibility = M2s[:, :, i]

        C2 = np.matmul(K2, M2_possibility)

        P, current_error = triangulate(C1, pts1, C2, pts2)

        z_vals = P[:, -1]
        # M should have only positive Z values and the lowest error to be saved
        if np.all(z_vals > 0) and current_error < lowest_error:
            best_M2 = i
            lowest_error = current_error

    # extract the best m2
    final_M2 = M2s[:, :, best_M2]
    C1_final = np.matmul(K1, M1)
    C2_final = np.matmul(K2, final_M2)
    P_final, error_final = triangulate(C1_final, pts1, C2_final, pts2)

    return final_M2, C2_final, P_final


if __name__ == "__main__":

    # Loading correspondences
    correspondence = np.load('data/some_corresp.npz')
    # Loading the intrinscis of the camera
    intrinsics = np.load('data/intrinsics.npz')
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)

    np.savez("results/q3_3.npz", M2=M2, C2=C2, P=P)

    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    assert (err < 500)
