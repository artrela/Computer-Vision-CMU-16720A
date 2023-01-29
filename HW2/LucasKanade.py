from operator import inv
from re import I
from tkinter import W
from cv2 import imshow
import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

# user imports
import scipy.ndimage
from PIL import Image
import matplotlib.pyplot as plt


def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """

    # Put your implementation here
    # set up the threshold
    ################### TODO Implement Lucas Kanade ###################

    p = np.copy(p0)
    deltaP = np.zeros(2)

    # define the jacobian
    jac = np.array([[1, 0], [0, 1]])

    # make a spline of It and It1
    # it, it1 dims
    It_row, It_col = It.shape
    It1_row, It1_col = It1.shape
    # arrange vals then make a mesh to create the spline
    It_x = np.arange(0, It_row)
    It_y = np.arange(0, It_col)
    It_spline = RectBivariateSpline(It_x, It_y, It)
    It1_x = np.arange(0, It1_row)
    It1_y = np.arange(0, It1_col)
    It1_spline = RectBivariateSpline(It1_x, It1_y, It1)

    # obtain the template from It spline
    # first define the coordinates of the rectangle
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]
    # define the meshgrid to obtain the template from
    rect_row = np.arange(y1, y2)
    rect_col = np.arange(x1, x2)
    rr_mesh, rc_mesh = np.meshgrid(rect_row, rect_col)
    T = It_spline.ev(rr_mesh, rc_mesh)

    # ? Debugging Log
    # // with open("lk_log.txt", "a") as log:
    # //     log.write(f"Start Values | Delta P {deltaP} | P: {str(p)}\n")

    error = 1
    i = 0
    while error > threshold and i < num_iters:
        # now find the warped image I(W(x;p))
        # the image is defined by its new x and y pos
        W_xp_row = np.arange(y1, y2) + p[1]
        W_xp_col = np.arange(x1, x2) + p[0]
        W_xp_r_mesh, W_xp_c_mesh = np.meshgrid(W_xp_row, W_xp_col)

        # evaluate the spline with the new meshes, and the derivatives
        I_Wxp = It1_spline.ev(W_xp_r_mesh, W_xp_c_mesh)
        Ix = It1_spline.ev(W_xp_r_mesh, W_xp_c_mesh, dx=1, dy=0)
        Iy = It1_spline.ev(W_xp_r_mesh, W_xp_c_mesh, dx=0, dy=1)

        # combine the derivatives into a tensor, the matrix A
        A = np.stack((Iy.flatten(), Ix.flatten()), axis=1)
        # mult with jacobian
        A = np.dot(A, jac)

        # find the error image B
        D = (T - I_Wxp).flatten()
        B = np.dot(A.T, D)

        # now solve the linear equations in steps
        Hessian = np.dot(A.T, A)
        invHessian = np.linalg.inv(Hessian)
        deltaP = np.dot(invHessian, B)

        p += deltaP
        i += 1

        # ? Debugging Log
        # //with open("lk_log.txt", "a") as log:
        # //     log.write(f"Iteration: {i} | Delta P {deltaP} | P: {str(p)}\n")

        error = np.linalg.norm(deltaP, ord=2)**2
        # //print(f"Iteration {i} || Error: {error} || P: {p}")

    # ? Debugging Log
    # // with open("lk_log.txt", "a") as log:
    # //     log.write(
    # //         f"Total Iterations: {i} | Final dp: {deltaP} | Final P: {str(p)}\n")

    # //print(f"Iteration {i} || Error: {error} || P: {p}")

    return p
