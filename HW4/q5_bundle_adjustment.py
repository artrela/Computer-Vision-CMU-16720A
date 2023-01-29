import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2

import scipy

# Insert your package here

import random as rd
import math
import scipy
from q4_2_visualize import compute3D_pts
from q3_2_triangulate import triangulate
# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""


def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:, 0], P_before[:, 1], P_before[:, 2], c='blue')
    ax.scatter(P_after[:, 0], P_after[:, 1], P_after[:, 2], c='red')
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on
        the results/expected number of inliners. You can also define your own metric.
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values

'''


def ransacF(pts1, pts2, M, nIters=500, tol=1):
    # Replace pass by your implementation

    # init vals
    N = pts1.shape[0]
    locs1 = pts1
    locs2 = pts2
    max_iters = nIters

    inliers = np.zeros(shape=(N, 1))

    for i in range(max_iters):

        # get 8 rand ind
        randInd = np.array(rd.sample(range(N), 8))
        rand_locs1 = locs1[randInd]
        rand_locs2 = locs2[randInd]

        # find temp fundametnal matrix
        tempF = eightpoint(rand_locs1, rand_locs2, M)  # change source/dest

        Hom_locs1 = np.hstack((locs1, np.ones(shape=(N, 1))))
        Hom_locs2 = np.hstack((locs2, np.ones(shape=(N, 1))))

        # find the error of the fundamental matrix
        temp_inliers = calc_epi_error(Hom_locs1, Hom_locs2, tempF)

        # find which inliers are within tolerance
        temp_inliers = temp_inliers < tol
        temp_inliers = temp_inliers.astype(np.int8)

        highestNumInliers = inliers[inliers == 1].shape[0]
        currNumInliers = temp_inliers[temp_inliers == 1].shape[0]

        # if this number of inliers is larger than before, make it the best one
        if currNumInliers > highestNumInliers:
            inliers = temp_inliers

    # Find inliers from inlier vector
    inlierInd = np.where(inliers == 1)
    inlierLocs1 = locs1[inlierInd]
    inlierLocs2 = locs2[inlierInd]

    # recalc F with inliers
    bestF = eightpoint(inlierLocs1, inlierLocs2, M)

    return bestF, inliers


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''


def rodrigues(r):
    # Replace pass by your implementation

    # https://courses.cs.duke.edu//fall13/compsci527/notes/rodrigues.pdf (page 5)

    I = np.diag([1, 1, 1])
    r = np.concatenate(r).T

    theta = np.linalg.norm(r)

    u = r/theta

    # skew symmetric matrix
    u_x = np.array([[0, -u[2], u[1]],
                    [u[2], 0, -u[0]],
                    [-u[1], u[0], 0]])

    R = I * math.cos(theta) + (1 - math.cos(theta)) * \
        u * u.T + u_x * math.sin(theta)

    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''


def invRodrigues(R):
    # Replace pass by your implementation

    # https://courses.cs.duke.edu//fall13/compsci527/notes/rodrigues.pdf (page 5)

    A = (R - R.T) / 2

    p = np.array([A[2, 1], A[0, 2], A[1, 0]]).T

    s = np.linalg.norm(p)

    c = (R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2

    u = p / s

    theta = np.arctan2(s, c)

    r = u * theta

    return r


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''


def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation

    # extract relevant information from the x vector
    rot_2 = x[-6:-3].reshape((3, 1))
    w_3D = x[0:-6].reshape((-1, 3))
    trans_2 = x[-3:].reshape((3, 1))

    w_3dstacked = np.hstack([w_3D, np.ones((w_3D.shape[0], 1))])

    # find big rotation matrix R
    R2 = rodrigues(rot_2)

    # assemble extrinsic matrix
    M2 = np.hstack((R2, trans_2))

    # Find the camera matrices
    C1 = np.matmul(K1, M1)
    C2 = np.matmul(K2, M2)

    # project 3d points back onto 2d plane
    p1_est = np.matmul(C1, w_3dstacked.T)
    p2_est = np.matmul(C2, w_3dstacked.T)

    # normalize
    p1_est /= p1_est[-1, :]
    p2_est /= p2_est[-1, :]

    # take only x, y
    p1_est = p1_est[0:2, :].T
    p2_est = p2_est[0:2, :].T

    # calculate the residuals
    residuals = np.concatenate(
        [(p1 - p1_est).reshape([-1]), (p2-p2_est).reshape([-1])])

    return residuals


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual.
        You can try different (method='..') in scipy.optimize.minimize for best results.
'''


def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    obj_start = obj_end = 0
    # ----- TODO -----
    # YOUR CODE HERE

    # get init for stating translation and rotation parts
    T2i = M2_init[:, 3:]
    R2i = M2_init[:, :3]
    # use inv rod
    R2i = invRodrigues(R2i)

    # prepack individ parts
    P_init_reshape = P_init[:, :3].reshape((-1, 1))
    R2i_reshape = R2i.reshape((-1, 1))

    # prepack function var to minmize
    x = np.concatenate([P_init_reshape, R2i_reshape, T2i])
    x = x.reshape((-1, 1))

    def f(x):
        r = np.sum(rodriguesResidual(K1, M1, p1, K2, p2, x) ** 2)
        return r

    t = scipy.optimize.minimize(f, x)
    f = t.x

    # extract and reshape the vals from function solution
    P, r2, t2 = f[:-6], f[-6:-3], f[-3:]
    P, r2, t2 = P.reshape((-1, 3)), r2.reshape((3, 1)), t2.reshape((3, 1))

    # get big R from little r
    R2 = rodrigues(r2).reshape((3, 3))
    # stack together to obtain the camera matrix
    M2 = np.hstack((R2, t2))

    obj_end = t.fun

    np.savez('results/q5.npz', f=f, M2=M2, P=P,
             obj_start=obj_start, obj_end=obj_end)

    return M2, P, obj_start, obj_end


if __name__ == "__main__":

    np.random.seed(1)  # Added for testing, can be commented out

    some_corresp_noisy = np.load(
        'data/some_corresp_noisy.npz')  # Loading correspondences
    # Loading the intrinscis of the camera
    intrinsics = np.load('data/intrinsics.npz')
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    noisy_pts1, noisy_pts2 = some_corresp_noisy['pts1'], some_corresp_noisy['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    F, inliers = ransacF(noisy_pts1, noisy_pts2,
                         M=np.max([*im1.shape, *im2.shape]))

    F_orig = eightpoint(noisy_pts1, noisy_pts2,
                        M=np.max([*im1.shape, *im2.shape]))

    # print(F)

    # YOUR CODE HERE
    displayEpipolarF(im1, im2, F_orig)
    displayEpipolarF(im1, im2, F_orig)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(
        noisy_pts1), toHomogenous(noisy_pts2)

    assert (F.shape == (3, 3))
    assert (F[2, 2] == 1)
    assert (np.linalg.matrix_rank(F) == 2)

    # YOUR CODE HERE
    inlier_ind = np.where(inliers == 1)
    inlier_pts1 = noisy_pts1[inlier_ind]
    inlier_pts2 = noisy_pts2[inlier_ind]

    print(f"\nNum Inliers: {inlier_ind[0].shape[0]}/{noisy_pts1.shape[0]}")

    M1 = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]])

    # # error = rodriguesResidual(K1, M1, inlier_pts1, K2, inlier_pts2, )
    # # print(error)

    M2, C2, P_before = findM2(F, inlier_pts1, inlier_pts2,
                              intrinsics, filename="q5.npy")

    # M1 = np.array([[1, 0, 0, 0],
    #                [0, 1, 0, 0],
    #                [0, 0, 1, 0]])

    # M2, P_after, obj_start, obj_end = bundleAdjustment(
    #     K1, M1, inlier_pts1, K2, M2, inlier_pts2, P_before)

    C1 = np.matmul(K1, M1)
    C2 = np.matmul(K2, M2)

    # _, error = triangulate(C1, inlier_pts1, C2, inlier_pts2)
    # print("\nReprojection Error: \n", round(error, 2))

    # plot_3D_dual(P_before, P_after)

    # Simple Tests to verify your implementation:
    # from scipy.spatial.transform import Rotation as sRot
    # rotVec = sRot.random()
    # mat = rodrigues(rotVec.as_rotvec())

    # assert (np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3)
    # assert (np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3)

    # YOUR CODE HERE
