import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize

# Insert your package here


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to usethe normalized points instead of the original points)
    (6) Unscale the fundamental matrix
'''


def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation

    N = pts1.shape[0]

    pts1 = np.hstack((pts1, np.ones(shape=(N, 1))))
    pts2 = np.hstack((pts2, np.ones(shape=(N, 1))))

    T = np.array([[(1/M), 0, 0],
                  [0, (1/M), 0],
                  [0, 0, 1]])

    # 1.
    pts1_N = np.matmul(T, pts1.T).T
    pts2_N = np.matmul(T, pts2.T).T

    # 2.
    A = []
    for i in range(N):

        x1, y1 = pts1_N[i, 0], pts1_N[i, 1]
        x2, y2 = pts2_N[i, 0], pts2_N[i, 1]

        A.append(np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]))

    A = np.vstack(A)

    # 3.
    U, S, vT = np.linalg.svd(A)

    v = vT.T
    f = v[:, -1]

    # 4.
    F = np.reshape(f, newshape=(3, 3)).T
    F = _singularize(F)

    # 5.
    F = refineF(F, pts1_N[:, 0:2], pts2_N[:, 0:2])

    # 6.
    F = np.matmul(np.transpose(T), np.matmul(F, T))
    F /= F[-1, -1]

    return F


if __name__ == "__main__":

    # Loading correspondences
    correspondence = np.load('data/some_corresp.npz')
    # Loading the intrinscis of the camera
    intrinsics = np.load('data/intrinsics.npz')
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    M = np.max([*im1.shape, *im2.shape])

    F = eightpoint(pts1, pts2, M)

    np.savez("results/q2_1.npz", F=F, M=M)

    # Q2.1
    print(f"\nRecovered F:\n\n{F}")
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert (F.shape == (3, 3))
    assert (F[2, 2] == 1)
    assert (np.linalg.matrix_rank(F) == 2)
    assert (np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)
