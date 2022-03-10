#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 22:22:54 2022

@author: katie
"""


import numpy as np
import math
import cv2 as cv

from skimage.color import rgb2gray
"""
Parameters
"""
W = 105 #must be odd
sigma = 1 #can be chosen between 0 and 1

#left and right images must be the same size

#image_L_grayscale = rgb2gray(image_L) 
#image_R_grayscale = rgb2gray(image_R) 
#results in a 2d image
"""
image.shape = 
"""

def create_all_submatrices(image, W): #we must have M <=min(image_L.shape)
    windows = []
    windows_indices = []
    for x in range(0,image.shape[0]-W):
        for y in range(0,image.shape[1]-W):
            window = image[x:(W+x),y:(W+y)]
            windows.append(window)
            windows_indices.append([x,y])
    return windows, windows_indices
    
def C_ij(A,B):
    A_mean = np.mean(A)
    B_mean = np.mean(B)
    A_sigma = np.std(A)
    B_sigma = np.std(B)
    numerator_list = []
    r_ij = np.linalg.norm(A-B)
    for u in range(0,W):
        for v in range(0,W):
            numerator_list.append((A[u,v] - A_mean)*(B[u,v] - B_mean))
    numerator = np.sum(numerator_list)
    denominator = (W**2)* A_sigma * B_sigma
    return numerator/denominator

def G_ij(A,B):
    C_ij_ = C_ij(A,B)
    r_ij = np.linalg.norm(A-B)
    return ((C_ij_ + 1)/2)*np.exp(-((r_ij**2)/(2*(sigma**2))))

def svd_stereo_matching(image_L, image_R):
    image_L = rgb2gray(image_L)
    image_R = rgb2gray(image_R)
    windows_L, windows_indices_L = create_all_submatrices(image_L, W)
    windows_R, windows_indices_R = create_all_submatrices(image_R, W)
    G = np.empty((len(windows_L),len(windows_R)))
    for i in range(0,len(windows_L)):
        for j in range(0,len(windows_R)):
            G_ij_ = G_ij(windows_L[i], windows_R[j])
            G[i,j] = G_ij_
            print("on i = " + str(i) + "and j = " + str(j))
    T, D, Uh = np.linalg.svd(G, full_matrices=True)
    E = D
    for i in range(0,E.shape[0]):
        for j in range(0,E.shape[1]):
            if E[i,j] >= 0:
                E[i,j]= 1
            
    P = T @ E @ Uh

    matches = []
    for i in range(0,P.shape[0]):
        for j in range(0,P.shape[1]):
            if P[i,j]== P[i,:].max() and P[i,j]== P[:,j].max():
                matches.append([i,j])
    
    matches_windows_indices_L_and_R = []
    for i in range(0,len(matches)):
        matches_windows_indices_L_and_R.append([windows_indices_L[matches[i][0]],windows_indices_R[matches[i][1]]])
    return matches_windows_indices_L_and_R #this is a list of tuples where the first element of the tuple is the match in the left image and the second element is the corresponding match in the right image

img1 = cv.imread('/Users/katie/Downloads/python-depthmaps-main/left_img.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('/Users/katie/Downloads/python-depthmaps-main/right_img.png', cv.IMREAD_GRAYSCALE)


matches_windows_indices_L_and_R = svd_stereo_matching(img1,img2)