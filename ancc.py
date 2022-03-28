#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 11:33:31 2022

@author: katie
"""


import numpy as np
import math
from skimage import io, color
import match as Match
import matplotlib.pyplot as plt

# =============================================================================
# sigma_d = 0
# sigma_s = 0
# lambda_ = 0
# V_max = 5
# m = 0
# M = (m,m)
# p_coordinate = (1,2)
# theta = 0
# 
# =============================================================================
"""
t should iterate through all pixels in a given window W(p)

All pixels should be of the shape (3,)

p

"""
def R_3(gamma, p, image,sigma_s,sigma_d, channel,width):
    window = W(p,width)
    inner_term = []
    omega_ = []
    for t in window:
        if Match.inRect(t,image.shape):
            try:
                a = w(p,t,sigma_d, sigma_s,channel,image)
                b = K(t,channel,image)
                omega_.append(a)
                inner_term.append(a*b)
            except:
                pass
    #omega_ = omega(p,image, width,sigma_d, sigma_s,channel)
    Z_ = np.mean(omega_)
    inner = inner_term/Z_
    R_3 = K(p,channel,image) - inner
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

def create_coordinate_list(p_coordinate,m):
    m_list = create_m_list(m)
    
    m_p_list = []
    for i in range(0,len(m_list)):
        m_p_list.append(list( map(add, m_list[i], p_coordinate) ))
    return(m_p_list)
            
def omega(p_coordinate,image,width,sigma_d, sigma_s,channel):
    coordinates_in_M = W(p_coordinate, width)

    omega = []
    for t in coordinates_in_M:
        if Match.inRect(t,image.shape):
            try:
                w_ = w(p_coordinate, t,sigma_d, sigma_s,channel,image)*K(t,channel,image)
                omega.append(w_)
            except:
                pass
    return omega

 

def Z(omega):
    Z = np.mean(omega)
    return Z
    
def K(t, color,image):
    term = (image[t[0],t[1],color])/((image[t[0],t[1],0]*image[t[0],t[1],1]*image[t[0],t[1],1])**(1/3))
    if term >= 0:
        K = math.log(term,10)
    else:
        K = 0
    return K

def I(t):
    mean = np.mean(t)
    return mean

def w(p, t, sigma_d, sigma_s,channel,image):
    p = np.asarray(p)
    t = np.asarray(t)
    I_p = I(image[p[0],p[1],:])
    I_t = I(image[t[0],t[1],:])
    term1 = -((np.linalg.norm( np.array([p[0],p[1]])-np.array([t[0],t[1]]) ))**2)
    term2= -((np.linalg.norm( I_p + I_t))**2)
    w = math.exp((term1+term2)/(2*(sigma_d**2)))
    return w

def ANCC(p_L, p_R, image_L, image_R, channel,gamma,width,sigma_d, sigma_s):
    list_of_coordinates_L = W(p_L,width)
    list_of_coordinates_R = W(p_R,width)
    numerator = []
    denominator_1 = []
    denominator_2 = []
    R_3_L = R_3(gamma, p_L, image_L, sigma_s,sigma_d, channel,width)
    R_3_R = R_3(gamma, p_R, image_R, sigma_s,sigma_d, channel,width)
    for i in range(0,len(list_of_coordinates_L)):
        if Match.inRect(list_of_coordinates_L[i], image_L.shape) and Match.inRect(list_of_coordinates_R[i], image_R.shape):
            w_L = w(p_L, list_of_coordinates_L[i],sigma_d, sigma_s,channel,image_L)
            w_R = w(p_R, list_of_coordinates_R[i], sigma_d, sigma_s, channel,image_R)
            term_1 = w_L*w_R*R_3_L*R_3_R
            numerator.append(term_1)
            left = np.sqrt((np.linalg.norm(w_L*R_3_L))**2)
            right = np.sqrt((np.linalg.norm(w_L*R_3_R))**2)
            denominator_1.append(left)
            denominator_2.append(right)
    numerator = np.sum(numerator)
    denominator_1 = np.sqrt(np.sum(denominator_1))
    denominator_2 = np.sqrt(np.sum(denominator_2))
    denominator = denominator_1 * denominator_2
    return numerator/denominator
 
def D(p_L, p_R, image_L, image_R, gamma, theta,width,sigma_d, sigma_s):
    # 0 = red, 1 = green, 2 = blue
    color_channels = [0,1,2]
    log_color_channels = [0,1,2]
    cielab_image_L = color.rgb2lab(image_L)
    cielab_image_R = color.rgb2lab(image_R)
    color_ANCC = []
    log_color_ANCC = [0]
    #for i in color_channels:
    #    color_ANCC.append(ANCC(p_L, p_R, image_L, image_R, channel = i,gamma=gamma, width=width,sigma_d = sigma_d, sigma_s = sigma_s))
    for i in log_color_channels:
        log_color_ANCC.append(ANCC(p_L, p_R, cielab_image_L, cielab_image_R, channel = i,gamma=gamma,width=width,sigma_d = sigma_d, sigma_s = sigma_s))
    D = 1 - ((theta*(np.sum(color_ANCC)))+(1-theta)*(np.sum(log_color_ANCC)))
    if math.isnan(D) == False:
        return D
    else:
        D = 0
        return D

def V_pq(f_p, f_q, lambda_ancc, V_max=5):
    return lambda_ancc * np.maximum((np.linalg.norm((f_p, f_q))**2), V_max)

def N(p):
    list_of_terms_1 = [-1,0,1]
    list_of_terms_2 = [-1,0,1]
    N = []
    for i in list_of_terms_1:
        for j in list_of_terms_2:
            N.append(list_of_terms.append([p[0]+i,p[1]+j]))
    return N

def W(p,m):
    window_indices = []
    for i in range(-m,m):
        for j in range(-m,m):
            window_indices.append([p[0]+i,p[1]+j])
    return window_indices

#def E_f_p(p_L, p_R, imageLeft, imageRight, parameters): #assume disparities is some matrix of f_p's
#    imageLeft_LAB = color.rgb2lab(imageLeft)
#    imageLeft_Right = color.rgb2lab(imageRight)
#    N = N(p)
#    V= []
#    D = []
#    
#    f_p = d2 - d1 
#        D.append(D(p_L,p_R, image_L, image_R, parameters.gamma, parameters.theta))
#        for dq in N:
#            f_q = dq - d1
#            V.append(V_pq(f_p, f_q, parameters.lambda_ancc, parameters.V_max))
#    E = np.sum(D) + np.sum(V)
#    return E
#        
    
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
    
    
