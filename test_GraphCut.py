#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:51:13 2022

@author: katie
"""

import match
import cv2

# main programme
def main(file1, file2, color, dispMin, dispMax):
    # load two images
    imgLeft = cv2.imread(file1)
    imgRight = cv2.imread(file2)
    #imgLeft = imgLeft[200:, 100:]
    #imgLeft = cv2.resize(imgLeft, (200,300) ,interpolation = cv2.INTER_AREA)
    #imgRight = cv2.resize(imgRight, (200,300) ,interpolation = cv2.INTER_AREA)
    
    # Default parameters
    K = -1
    lambda_ = -1
    lambda1 = -1
    lambda2 = -1
    params = match.Parameters(is_L2=True,
                              denominator=1,
                              edgeThresh=8,
                              lambda1=lambda1,
                              lambda2=lambda2,
                              K=K,
                              maxIter=10,
                              sigma_d = 14,
                              sigma_s = 3.8,
                              lambda_ancc = 1/30,
                              V_max = 16,
                              width = 3,
                              theta = 0.7,
                              gamma = 1
                              )

    # create match instance
    m = match.Match(imgLeft, imgRight, color)
    m.SetDispRange(dispMin, dispMax)
    m = match.fix_parameters(m, params, K, lambda_, lambda1, lambda2)
    m.kolmogorov_zabih()
    m.saveDisparity("./results/disparity1.jpg")


if __name__ == '__main__':
    filename_left = "./images/perra_7.jpg"
    filename_right = "./images/perra_8.jpg"
    is_color = True
    disMin = -16
    disMax = 16
    main(filename_left, filename_right, is_color, disMin, disMax)
    
