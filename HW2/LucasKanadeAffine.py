import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import sobel
from scipy.ndimage import affine_transform
import cv2
import matplotlib.pyplot as plt


def disp_img(img, heading):
    img = np.array(img)
    cv2.imshow(heading, img)
    cv2.waitKey()


def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ################### TODO Implement Lucas Kanade Affine ###################

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
        1.e Convert the meshgrid into a flattened array
        1.g Precompute the unwarpe template and gradients of It1"""

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
    xInd1 = rows_mesh1.flatten()
    yInd1 = cols_mesh1.flatten()

    #! 1.f
    arr_1s = np.ones(shape=(1, len(xInd1)))
    # origPixels = np.vstack((xInd1, yInd1, arr_1s))

    #! 1.g
    Ix = It1_spline.ev(rows_mesh1, cols_mesh1, dx=1, dy=0)
    Iy = It1_spline.ev(rows_mesh1, cols_mesh1, dx=0, dy=1)

    It1_frame = It1_spline.ev(rows_mesh1, cols_mesh1).T
    template = It_spline.ev(rows_mesh0, cols_mesh0).T
    # // disp_img(template, "template")

    # init error and iterations
    error = 1
    i = 0
    p = np.zeros(6)
    while error > threshold and i < num_iters:

        M[0, 0] = 1 + p[0]
        M[0, 1] = p[1]
        M[0, 2] = p[2]
        M[1, 0] = p[3]
        M[1, 1] = p[4] + 1
        M[1, 2] = p[5]

        # print("i: ", i, "\n", "M: ", M)

        """ Step 1: Affine warp It1, and the It1_grad  """

        # print("Step 1")

        warpedIt1 = affine_transform(It1_frame, M)
        # // disp_img(warpedIt1, "warped it1")
        warpedIx = affine_transform(Ix, M).T
        #disp_img(warpedIx, "warped ix")
        warpedIy = affine_transform(Iy, M).T
        #disp_img(warpedIy, "warped iy")

        # flatten the gradients and warped image
        warpedIt1_flat = warpedIt1.flatten()
        warpedIx_flat = warpedIx.flatten()
        warpedIy_flat = warpedIy.flatten()

        """Step 2: Fill in where 0s exist on the warped image """

        # print("Step 2")

        template_temp = np.copy(template)
        zero_ind = np.where(warpedIt1 == 0)
        template_temp[zero_ind] = 0
        # // disp_img(template, "template blacked out")
        template_temp_flat = template_temp.flatten()

        """Step 3: Find the error image """

        # print("Step 3")

        errorImg = (template_temp_flat - warpedIt1_flat)

        # print("Error Image\n", errorImg[20:30])

        """Step 4: Calculate the steepest descent directly  """

        # print("Step 4")

        # flatten the rows of the meshgrid, to signify the repeated x, y
        x_locs = xInd1.flatten()
        y_locs = yInd1.flatten()

        # find the elements of the steepest descent beforehand
        ele1 = warpedIx_flat * x_locs
        ele2 = warpedIy_flat * x_locs
        ele3 = warpedIx_flat * y_locs
        ele4 = warpedIy_flat * y_locs
        ele5 = warpedIx_flat
        ele6 = warpedIy_flat

        steepestDescent = np.array([ele2, ele4, ele6, ele1, ele3, ele5]).T

        #print("\n\nsteepest D\n", steepestDescent[0:10], "\n\n")

        """Step 5: use the steepest descent to find the hessian, hessian inverse"""

        # print("Step 5")
        hessian = np.dot(steepestDescent.T, steepestDescent)
        invHessian = np.linalg.inv(hessian)

        """Step 6: calculate delta p, update params"""

        # print("Step 6")
        deltaP = np.dot(invHessian, np.dot(steepestDescent.T, errorImg))

        p += deltaP

        #print("M\n", M)
        error = np.linalg.norm(deltaP)

        i += 1
    #
    print("Final Iterations: ", i)

    return M

# #total_pixels = It1.shape[0] * It1.shape[1]

#     # jacobian = np.zeros(shape=(2, 6, total_pixels))

#     # j = 0
#     # for row in range(0, It.shape[0]):
#     #     for column in (0, It.shape[1]):
#     #         jacobian[:, :, j] = np.array(
#     #             [[row, column, 1, 0, 0, 0], [0, 0, 0, row, column, 1]])
#     #         # [[row, 0, column, 0, 1, 0], [0, row, 0, column, 0, 1]])
#     #         j += 1
#     w, h = It.shape
#     Ix = sobel(It1, axis=0)
#     Iy = sobel(It1, axis=1)
#     x = np.linspace(0, w)
#     y = np.linspace(0, h)
#     x_mesh, y_mesh = np.meshgrid(x, y)
#     x_mesh, y_mesh = x_mesh.flatten(), y_mesh.flatten()
#     print(x_mesh.shape)

#     i = 0
#     error = 1
#     while error > threshold and i < num_iters:

#         Ix_warped = affine_transform(Ix, M)  # w, h
#         Iy_warped = affine_transform(Iy, M)
#         Ix_warped = Ix_warped.reshape(1, w*h)  # 1, wh
#         Iy_warped = Iy_warped.reshape(1, w*h)  # 1, wh

#         It1_warped = affine_transform(It1, M)
#         A1 = Ix_warped*x_mesh
#         A2 = Ix_warped*y_mesh
#         A3 = Iy_warped*x_mesh
#         A4 = Iy_warped*y_mesh

#         A = np.concatenate((A1, A2, Ix_warped.T, A3, A4, Iy_warped.T), axis=0)
#         # plt.imshow(It1_warped)
#         # plt.show()

#         # Calculate the gradients of the warped frame

#         # template_temp = np.copy(It)

#         # zero_ind = np.where(It1_warped == 0)

#         # template_temp[zero_ind] = 0

#         # find the error image B
#         # A = np.transpose(
#         #     np.stack((Ix_warped.flatten(), Iy_warped.flatten()), axis=1))
#         # A = A[:, :, np.newaxis]

#         D = (It1_warped - It).flatten()

#         # gDescent = np.zeros((6, total_pixels))
#         # for k in range(0, total_pixels):
#         #     A_temp = A[:, k]
#         #     jac_temp = jacobian[:, :, k]
#         #     gDescent[:, k] = np.matmul(A_temp.T, jac_temp)

#         # mult with jacobian
#         # gDescent_3dim = np.einsum("abc, adb->bcd", A, jacobian)
#         # gDescent_2dim = gDescent_3dim[:, 0, :]

#         # b = np.dot(gDescent_2dim.T, D)
#         b = np.dot(A.T, D)
#         # b = np.matmul(gDescent, D)

#         # find dp
#         # Hessian = np.dot(gDescent_2dim.T, gDescent_2dim)
#         Hessian = np.dot(A.T, A)
#         # Hessian = np.dot(gDescent.T, gDescent)
#         invHessian = np.linalg.pinv(Hessian)
#         deltaP = np.dot(invHessian, b)

#         M[0][0] += deltaP[0]
#         M[0][1] += deltaP[1]
#         M[0][2] += deltaP[2]
#         M[1][0] += deltaP[3]
#         M[1][1] += deltaP[4]
#         M[1][2] += deltaP[5]
