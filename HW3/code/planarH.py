import numpy as np
import cv2
from sklearn.metrics import max_error
import tqdm
import random as rd


def orderMatches(matches, locs1, locs2):
    """
    Partner function for matchPics.py. Takes array of interest points locs1, locs2 and returns 

    Steps:
    1. Init return value
    2. [For Each Match Pair] Get the match from locs1 that maps to locs2 with matches. 
        - The i-th row of matches gives [locs1_index, locs2_index]
        - use the locs1_index to get a val from locs1, and the same idea for locs2
        - make sure you get that row of locs but both columns
    3. Append the new match to the list
    4. Turn to a np.array by vertically stacking so that we can get (N, 2) shape

    Args:
        matches (np.array): Nx2 Matrix that refers to which index from locs1 maps to which index from locs2 [index2, index2]
        locs1 (np.array): Mx2 Interest points in source image
        locs2 (np.array): Mx2 Interest points in destination image

    Returns:
        x1 (np.array): ordered, matched interest points of image 1
        x2 (np.array): ordered, matched interest points of image 2

    """

    # print("Order the Matches...\n")

    # 1.
    x1 = []
    x2 = []

    for i in range(matches.shape[0]):

        # 2.
        match1 = locs1[matches[i, 0], :]
        match2 = locs2[matches[i, 1], :]

        # 3.
        x1.append(match1)
        x2.append(match2)

    # 4.
    x1 = np.vstack(x1)
    x2 = np.vstack(x2)

    return x1, x2


def computeH(xSource, xDes):
    # Q2.2.1
    # Compute the homography between two sets of points
    """
    Args:
        X1 (np.array): Points of interest in the source image
        X2 (np.array): Points of interest in the desination image

    Returns:
        H2to1: Homography matrix from source to destination.
    """
    #? ------- Steps ----------?#
    """
    1. Init the A matrix (2n x 9), where n is the total number of points we have
    [For Each Point Pair]
    2. Find the x, y values from the source and destination image
    3. Each point pair has the following two rows:

    a. [x2, y2, 1, 0, 0, 0, -x1x2 , -x1y2, -x1] 
        [0, 0, 0, x2, y2, 1, -y1x2, -y1y2, -y1]

    4. Append to A matrix   
    5. Vstack all match pairs
    6. Perform SVD on the A matrix to get U, Sigma, V.T
    7. Take the last column of the V.T matrix, which should correspond to the eigenvectors with the 
    eigenvalue of (ideally) 0. The nontrival solution to Ah=0.
        - SVD gives v.T, and v is the eigvecs of A.T * A. 
        - Transpose v, then find the last column
    """

    # 1
    n = len(xSource)
    A = []

    for i in range(n):

        # 2
        x1, y1 = xSource[i, 0], xSource[i, 1]
        x2, y2 = xDes[i, 0], xDes[i, 1]

        # 3
        pairCols = np.array([[x2, y2, 1, 0, 0, 0, -x1*x2, -x1*y2, -x1],
                             [0, 0, 0, x2, y2, 1, -y1*x2, -y1*y2, -y1]])

        # 4
        A.append(pairCols)

    # 5
    A = np.vstack(A)

    # 6
    u, s, vT = np.linalg.svd(A)

    # 7
    v = vT.T
    h = v[:, -1]

    # 8
    H2to1 = np.reshape(h, newshape=(3, 3))

    return H2to1


def computeH_norm(x1, x2):
    # Q2.2.2
    """ Computes the homography, but with normalized match points.

    Args:
        X1 (np.array): Points of interest in the source image
        X2 (np.array): Points of interest in the destination image

    Returns:
        H2to1: Homography matrix from source to destination.
    """
    #? ------- Steps ------?#
    """
    Steps:
    1. Compute the mean, or centroid of the x1, x2 points
        - sum along all rows, divide by number of rows
        - divide by total num of points
    2. Subtract the centroid from all values of x1, x2 
        - leverage broadcasting
    3. Find the average scale
        a. average distance of each vector normalized to sqrt(2)
        b. max dist is sqrt(2)
    4. Multiply points by the mean
    5. Generate similarity transforms T1, T2
    6. Compute the normalized homography
    7. Denormalize the homography with T1 and T2    
    """

    # 1
    centroid1 = np.sum(x1, axis=0) / x1.shape[0]
    centroid2 = np.sum(x2, axis=0) / x2.shape[0]

    # 2
    x1_N = x1 - centroid1
    x2_N = x2 - centroid2

    # 3
    scale1 = np.sqrt(
        2) / np.max(np.linalg.norm(x1_N, axis=1), axis=0)
    scale2 = np.sqrt(
        2) / np.max(np.linalg.norm(x2_N, axis=1), axis=0)

    # 4
    x1_N *= scale1
    x2_N *= scale2

    # 5
    T1 = np.array([[scale1, 0, (-scale1*centroid1)[0]],
                   [0, scale1, (-scale1*centroid1)[1]],
                   [0, 0, 1]])

    T2 = np.array([[scale2, 0, (-scale2*centroid2)[0]],
                   [0, scale2, (-scale2*centroid2)[1]],
                   [0, 0, 1]])

    # 6
    H = computeH(x1_N, x2_N)

    # 7
    invT1 = np.linalg.inv(T1)
    H2to1 = np.matmul(invT1, np.matmul(H, T2))

    return H2to1


def computeH_ransac(locs1, locs2, opts):
    """ Using RANSAC to compute the best homography between source and destination. 

    Args:
        locs1 (np.array): Ordered list of matched points within the source image (Nx2).
        locs2 (np.array): Ordered list of matched points within the destination image (Nx2).
        opts (Namespace): Holds the parameters for RANSAC: mat_inters to run RANSAC for and the error threhold to 
                        to determine inliers.

    Returns:
        bestH2to1 (np.array): Best homography (3x3) calculated from inliners obtained from max_iters iterations of RANSAC. 
    """
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    # the tolerance value for considering a point to be an inlier
    inlier_tol = opts.inlier_tol

    # print("Performing RANSAC to compute the homography...\n")

    """_
    1. Initialize returns from the function
        - N -> how many match points we have
        - H -> 3x3
        - inliers -> Nx2
    [Enter for loop for max_iters]
    2. Select 4 random match pairs from locs1, and locs2
        - get four random indices from 0, N
        - use those four indices to take from locs1, locs2
    3. Use the locs1 to find the normalized Homography
    4. Find inliers
        4.1 Create homogeneous cords of locs1, locs2
            - Nx2 -> Nx3
            - Dimensionality match with H (Nx3)
        4.2 Get the estimate for locs2 by passing locs1 through H
            - Transpose homogeneous locs1 to multiply correct dimensions
                - (3x3) x (3xN)
        4.3 Normalize estimate of locs2
            - [lambda*x, lambda*y, lambda] / lambda -> [x, y, 1] 
            - Transpose the outcome again to get Nx3
        4.4 
            - Take the norm of homogenous locs2 - estimate of locs 1
            - if this norm is within the error then update that inlier 
        
        - if the norm from locs2 and estimated locs2 is within the error, it is an inlier
    5. If the temp inliers is greater than the current largest number of inliers, replace inliers
        - get current highest number of inliers
        - get the temporary value of inliers
    6. If the temporary amount of inliers is higher than the best amount of inliers, reset inliers
    [Exit the Loop]
    7. Use inliers list, get the inliers from locs1 and locs2
    8. Use the inliers to compute the new Homography
    """

    # 1.
    N = locs2.shape[0]
    inliers = np.zeros(shape=(N, 1))

    # tqdm.tqdm(range(max_iters), desc="Running RANSAC: "):
    for i in range(max_iters):

        # 2.
        # randInd = np.random.randint(0, N, size=4)
        randInd = np.array(rd.sample(range(N), 4))
        rand_locs1 = locs1[randInd]
        rand_locs2 = locs2[randInd]

        # 3.
        tempH2to1 = computeH_norm(rand_locs1, rand_locs2)  # change source/dest

        # 4.
        # 4.1
        Hom_locs1 = np.hstack((locs1, np.ones(shape=(N, 1))))
        Hom_locs2 = np.hstack((locs2, np.ones(shape=(N, 1))))
        # 4.2
        estimated_locs1 = np.dot(tempH2to1, Hom_locs2.T)
        # 4.3
        eLocs1Norm = (estimated_locs1 / estimated_locs1[2, :]).T
        # 4.4
        temp_inliers = np.linalg.norm(
            (Hom_locs1 - eLocs1Norm), axis=1)
        temp_inliers = temp_inliers < inlier_tol
        temp_inliers = temp_inliers.astype(np.int8)

        # 5.
        highestNumInliers = inliers[inliers == 1].shape[0]
        currNumInliers = temp_inliers[temp_inliers == 1].shape[0]

        # 6.
        if currNumInliers > highestNumInliers:
            inliers = temp_inliers

    #! Debug
    print(f"Highest Number of Inliers: {inliers[inliers == 1].shape[0]}")

    # 7.
    inlierInd = np.where(inliers == 1)
    inlierLocs1 = locs1[inlierInd]
    inlierLocs2 = locs2[inlierInd]

    # 8.
    bestH2to1 = computeH_norm(inlierLocs1, inlierLocs2)

    return bestH2to1, inliers


def compositeH(H2to1, source, destination):
    """Shows a warp from the source to the destination. 

    Args:
        H (np.array): 3x3 homography matrix 
        source (np.array): source image to be mapped to the destination
        destination (np.array): dest image to be mapped to
    """

    # find the inverse for 2 to 1
    # print("Inverse the homography H2to1 -> H1to2 ...")
    H = np.linalg.inv(H2to1)

    # warp view
    warpedImg = cv2.warpPerspective(
        source, H, dsize=(destination.shape[1], destination.shape[0]))

    # find where the warped image is equal to 0
    zeroInd = np.where(warpedImg == 0)
    # replace the 0 values from the warped with values from the destination
    warpedImg[zeroInd] = destination[zeroInd]

    return warpedImg
