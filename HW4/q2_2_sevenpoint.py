import numpy as np
import matplotlib.pyplot as plt
import sympy as sym


from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize

# Insert your package here


'''
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Sovling this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
'''


def sevenpoint(pts1, pts2, M):

    Farray = []
    # ----- TODO -----
    # YOUR CODE HERE
    N = pts1.shape[0]

    pts1 = np.hstack((pts1, np.ones(shape=(N, 1))))
    pts2 = np.hstack((pts2, np.ones(shape=(N, 1))))

    T = np.array([[(1/M), 0, 0],
                  [0, (1/M), 0],
                  [0, 0, 1]])

    pts1_N = np.matmul(T, pts1.T).T
    pts2_N = np.matmul(T, pts2.T).T

    A = []
    for i in range(N):

        x1, y1 = pts1_N[i, 0], pts1_N[i, 1]
        x2, y2 = pts2_N[i, 0], pts2_N[i, 1]

        A.append(np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]))

    A = np.vstack(A)

    _, _, v = np.linalg.svd(A)
    #! Everything up to here has been the same

    f1 = v.T[:, -1]
    f2 = v.T[:, -2]

    f1 = np.reshape(f1, newshape=(3, 3))
    f1 /= f1[-1, -1]
    f2 = np.reshape(f2, newshape=(3, 3))
    f2 /= f2[-1, -1]

    # Solve for the polynomial det9alpha*f1 + (1-alpha)*f2) = 0
    # symbolically
    alpha = sym.symbols("alpha")
    funct = alpha*f1 + (1 - alpha)*f2
    funct = sym.Matrix(funct)

    determinate = funct.det()

    coeff_0 = determinate.coeff(alpha, 0)
    coeff_1 = determinate.coeff(alpha, 1)
    coeff_2 = determinate.coeff(alpha, 2)
    coeff_3 = determinate.coeff(alpha, 3)

    polynomial = np.asarray(
        [coeff_0, coeff_1, coeff_2, coeff_3], dtype=float)

    poly_solution = np.polynomial.polynomial.polyroots(polynomial)

    # some of the solutions may have non real values so don't cound those
    for solution in poly_solution:

        if np.isreal(solution):
            F = solution*f1 + (1 - solution)*f2
            F = np.matmul(np.transpose(T), np.matmul(F, T))
            F /= F[-1, -1]

            Farray.append(F)

    return Farray


if __name__ == "__main__":

    # Loading correspondences
    correspondence = np.load('data/some_corresp.npz')
    # Loading the intrinscis of the camera
    intrinsics = np.load('data/intrinsics.npz')
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)

    F = Farray[0]

    print(f"\nRecovered F:\n\n{F}")

    np.savez('results/q2_2.npz', F, M, pts1, pts2)

    # fundamental matrix must have rank 2!
    # assert(np.linalg.matrix_rank(F) == 2)
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution.
    np.random.seed(1)  # Added for testing, can be commented out

    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M = np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo, pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))

    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    print("Error:", ress[min_idx])

    assert (F.shape == (3, 3))
    assert (F[2, 2] == 1)
    assert (np.linalg.matrix_rank(F) == 2)
    assert (np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)
