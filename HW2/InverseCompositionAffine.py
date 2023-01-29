import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2


def disp_img(img, heading):
    img = np.array(img)
    cv2.imshow(heading, img)
    cv2.waitKey()


def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    ################### TODO Implement Inverse Composition Affine ###################

    """" Precomputation goals:
    
    1. Create a spline that represents It, It1
        1.a Find the dims of the template, image
            - 1.a.1 It dims
            - 1.a.2 It1 dims
        1.b Find arrange axis from 0->w, 0->h for for both It, It1
            - 1.b.1 It axes
            - 1.b.2 It1 axes
        1.c Find the meshgrid that defines axes defined in 1.b
            - Will be used to evaluate the splines later on
        1.d Create the splines with the axes from 1.b and It, It1
        1.e Convert the meshgrid into a flattened array for x and y ind
        1.f Evaluate the spline
        1.g Precompute the unwarped template and gradients of It as well as It1"""

    #! 1.a
    x0, y0 = It.shape  # 1.a.1
    x1, y1 = It1.shape  # 1.a.2

    #! 1.b
    rows0 = np.arange(0, x0)  # 1.b.1
    cols0 = np.arange(0, y0)
    # ----------------------
    rows1 = np.arange(0, x1)  # 1.b.2
    cols1 = np.arange(0, y1)

    #! 1.c
    rows_mesh0, cols_mesh0 = np.meshgrid(rows0, cols0)
    rows_mesh1, cols_mesh1 = np.meshgrid(rows1, cols1)

    #! 1.d
    It_spline = RectBivariateSpline(rows0, cols0, It)
    It1_spline = RectBivariateSpline(rows1, cols1, It1)

    #! 1.e
    xInd0 = rows_mesh0.flatten()
    yInd0 = cols_mesh0.flatten()

    #! 1.f
    Ix_T = It_spline.ev(rows_mesh0, cols_mesh0, dx=1, dy=0)
    Iy_T = It_spline.ev(rows_mesh0, cols_mesh0, dx=0, dy=1)

    Ix_T_flat = Ix_T.flatten()
    Iy_T_flat = Iy_T.flatten()

    It1_frame = It1_spline.ev(rows_mesh1, cols_mesh1).T
    template = It_spline.ev(rows_mesh0, cols_mesh0).T

    #! 1.g
    # flatten the rows of the meshgrid, to signify the repeated x, y
    x_locs = xInd0.flatten()
    y_locs = yInd0.flatten()

    # find the elements of the steepest descent beforehand
    ele1 = Ix_T_flat * x_locs

    ele2 = Iy_T_flat * x_locs

    ele3 = Ix_T_flat * y_locs

    ele4 = Iy_T_flat * y_locs

    ele5 = Ix_T_flat

    ele6 = Iy_T_flat

    steepestDescent = np.array([ele2, ele4, ele6, ele1, ele3, ele5]).T

    #! 1.h
    hessian = np.dot(steepestDescent.T, steepestDescent)
    invHessian = np.linalg.inv(hessian)

    # init loop params
    i = 0
    error = 1
    deltaM = np.eye(3)
    deltaP = np.zeros(6)

    while error > threshold and i < num_iters:

        """ Warp It1 with the inverse of M0, M0 is the direction to go from template > It1
        """

        deltaM[0, 0] = 1 + deltaP[0]
        deltaM[0, 1] = deltaP[1]
        deltaM[0, 2] = deltaP[2]
        deltaM[1, 0] = deltaP[3]
        deltaM[1, 1] = deltaP[4] + 1
        deltaM[1, 2] = deltaP[5]

        M0 = np.dot(M0, np.linalg.inv(deltaM))

        warped_It1 = affine_transform(It1_frame, M0)

        # disp_img(template, "template")
        # disp_img(warped_It1, "It1 Frame")

        warped_It1_flat = warped_It1.flatten()

        """ Zero out out of bounds vals"""

        template_temp = np.copy(template)
        zero_ind = np.where(warped_It1 == 0)
        template_temp[zero_ind] = 0
        # // disp_img(template, "template blacked out")
        template_temp_flat = template_temp.flatten()

        """ calc the error image """

        errorImg = warped_It1_flat - template_temp_flat

        """ calculate dp  """

        deltaP = np.dot(invHessian, np.dot(steepestDescent.T, errorImg))

        # p += deltaP

        """ go directly off of delta p 
        make mtemp 1 = dleta p 0 
        m , inv mfinal  = matmul(M, inv(m_temp))"""

        error = np.linalg.norm(deltaP)

        i += 1

    print("Final Iterations: ", i)

    return M0  # invM0
