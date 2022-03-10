#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 11:33:31 2022

@author: katie
"""

import numpy as np
import math
from skimage import io, color


sigma_d = 0
sigma_s = 0
lambda_ = 0
V_max = 5
m = 0
M = (m,m)
p_coordinate = (1,2)
theta = 0

"""
t should iterate through all pixels in a given window W(p)

All pixels should be of the shape (3,)

p

"""
def R_3(t, gamma, p, image):
    W = create_coordinate_list(p)
    inner_term = []
    for t in W:
        inner_term.append(w(t)*K(t))
    inner = inner_term/Z
    omega = omega(p)
    Z = Z(omega)
    R_3 = gamma * (K(t) - inner)
    return R_3
"""
color: may be 0, 1, or 2 (red = 0, green = 1, blue = 2)
"""

def create_m_list(m):
#    M_x = np.arange(0,m,1)
#    M_y = np.arange(0,m,1)
    m_list = []
    for i in range(0,m):
        for j in range(0,m):
            m_list.append([i,j])
    return m_list

def create_coordinate_list(p_coordinate):
    m_list = create_m_list(m)
    
    m_p_list = []
    for i in range(0,len(m_list)):
        m_p_list.append(list( map(add, m_list[i], p_coordinate) ))
    return(m_p_list)
            
def omega(p, p_coordinate,image):
    coordinates_in_M = create_coordinate_list(p_coordinate)

    omega = []
    for i in coordinates_in_M:
        t = image[coordinates_in_M[0],coordinates_in_M[1],:]
        w = w(p, t)
        omega.append(w)
    return omega

def v(p, p_coordinate, image):
    coordinates_in_M = create_coordinate_list(p_coordinate)
    
    v = []
    for i in coordinates_in_M:
        t = image[coordinates_in_M[0],coordinates_in_M[1],:]
        R_3 = R_3(t)
        v.append(R_3)
    return v

def Z(omega):
    Z = np.mean(omega)
    return Z
    
def K(t, color):
    K = math.log((t[color])/((t[0]*t[1]*t[1])**(1/3)))
    return K

def I(t):
    mean = np.mean(t)
    return mean

def w(p, t):
    I_p = I(p)
    I_t = I(t)
    w = math.exp(-((np.linalg.norm(p-t))/(2*(sigma_d**2)))-((np.linalg.norm(I_p-I_t))/(2*(sigma_s**2))))
    return w

def ANCC_log(p_L, p_R, image_L, image_R, channel,gamma=gamma):
    list_of_coordinates_L = create_coordinate_list(p_L)
    list_of_coordinates_R = create_coordinate_list(p_R)
    numerator = []
    denominator_1 = []
    denominator_2 = []
    for i in len(list_of_coordinates):
        w_L = w(image_L[p_L], image_L[list_of_coordinates_L[i],channel])
        w_R = w(image_R[p_R], image_R[list_of_coordinates_R[i], channel)
        R_3_L = R_3(list_of_coordinates_L[i], gamma, p_L, image_L)
        R_3_R = R_3(list_of_coordinates_R[i], gamma, p_R, image_R)
        term_1 = w_L*w_R*R_3_L*R_3_R
        numerator.append(term_1)
        left = np.sqrt((np.norm(w_L*R_3_L))**2)
        right = np.sqrt((np.norm(w_L*R_3_R))**2)
        denominator_1.append(left)
        denominator_2.append(right)
    numerator = np.sum(numerator)
    denominator_1 = np.sqrt(np.sum(denominator_1))
    denominator_2 = np.sqrt(np.sum(denominator_2))
    denominator = denominator_1 * denominator_2
    f_p = p_R[1] - p_L[1]
    return f_p, numerator/denominator

def D(p_L, p_R, image_L, image_R, channel,gamma=gamma, theta=theta):
    # 0 = red, 1 = green, 2 = blue
    color_channels = [0,1,2]
    log_color_channels = [0,1,2]
    cielab_image_L = color.rgb2lab(image_L)
    cielab_image_R = color.rgb2lab(image_R)
    color_ANCC = []
    log_color_ANCC = []
    for i in color_channels:
        color_ANCC.append(ANCC(p_L, p_R, image_L, image_R, channel = i,gamma=gamma))
    for i in log_color_channels:
        log_color_ANCC.append(ANCC_log(p_L, p_R, cielab_image_L, cielab_image_R, channel = i,gamma=gamma))
    D = 1 - ((theta*(np.sum(color_ANCC)))+(1-theta)*(np.sum(log_color_ANCC)))
    return D

def V_pq(f_p, f_q, lambda_, V_max=5)
    return lambda_ * np.maximum((np.norm(f_p, f_q)**2), V_max)

def N(p):
    list_of_terms_1 = [-1,0,1]
    list_of_terms_2 = [-1,0,1]
    N = []
    for i in list_of_terms_1:
        for j in list_of_terms_2:
            N.append(list_of_terms.append([p[0]+i,p[1]+j]))
    return N

def W(p):
    return []

def E(disparities): #assume disparities is some matrix of f_p's
    
    N = N(p)
    V= []
    D = []
    
    for f_p in disparities:
        D.append(D()
        for q in N:
            f_q = disparities[q]
            V.append(V_pq(f_p, f_q, lambda_, V_max=5))
    E = 
        
    
def match(file1, file2, color, dispMin, dispMax):
    # load two images
    imgLeft = cv2.imread(file1)
    imgRight = cv2.imread(file2)

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
                              maxIter=4)

    # create match instance
    m = match.Match(imgLeft, imgRight, color)

    m.SetDispRange(dispMin, dispMax)
    m = match.fix_parameters(m, params, K, lambda_, lambda1, lambda2)
    disparities = m.kolmogorov_zabih()
    return disparities
    
def find_map(image_L, image_R, dispMin, dispMax):
    disparities = match(image_L, image_R, dispMin, dispMax)
    energy = E(disparities)
    
    return disparities, energy


        

#def obtain_ANCC_info(p_L, p_R, image_L, image_R):
    
    
#def ANCC(p):
    