#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for computing geostrophic wind speed
       1  dp
u = - --- --
       ρf dy

p = p_a + p_b * cos(yπ/L)

@author: andrea1994
"""

from __future__ import division
import numpy as np, matplotlib.pyplot as plt
    
def geostrophic_wind(rho=1.0, p_a=1e5, p_b=200.0, f=1e-4, L=2.4e6, y_min=0.0, \
                     y_max=1e6, N=1e5):
    # first, the arguments given to the function are tested
    if N<=0:
        raise ValueError('Error in geostrophic_wind: Argument N to \
                         geostrophic_wind should be > 0')
    if not(int(N) == N):
        raise ValueError('Error in geostrophic_wind: Argument N to \
                        geostrophic_wind should be an integer')
    if not(isinstance(float(rho),float) and float(rho) > 0):
        raise TypeError('Error in geostrophic_wind: Argument rho to \
                        geostrophic_wind should be a positive float')
    if not(isinstance(float(p_a),float) and float(p_a) > 0):
        raise TypeError('Error in geostrophic_wind: Argument rho to \
                        geostrophic_wind should be a positive float')
    if not(isinstance(float(p_b),float) and float(p_b) > 0):
        raise TypeError('Error in geostrophic_wind: Argument rho to \
                        geostrophic_wind should be a positive float')
    if p_a<p_b:
        raise ValueError('Error in geostrophic_wind: Argument p_b to\
                         geostrophic_wind should be smaller than p_a')
    if not(isinstance(float(f),float) and float(f) > 0):
        raise TypeError('Error in geostrophic_wind: Argument f to \
                        geostrophic_wind should be a positive float')
    if not(isinstance(float(L),float) and float(L) > 0):
        raise TypeError('Error in geostrophic_wind: Argument L to \
                        geostrophic_wind should be a positive float')
    if not(isinstance(float(y_min),float) and y_max>y_min):
        raise TypeError('Error in geostrophic_wind: Argument y_min to \
                        geostrophic_wind should be a float and smaller\
                        than y_max')
    if not(isinstance(float(y_max),float) and y_min<y_max):
        raise TypeError('Error in geostrophic_wind: Argument y_max to \
                        geostrophic_wind should be a float and greater\
                        than y_min')

    
    # conversion of N to an integer, if int(N) != N then TypeError is raised
    N = int(N)
    
    # initialisation of the y array
    y = np.zeros(N+1)
    Delta_y=(y_max - y_min)/N
    for i in range(N+1):
        y[i] = y_min + Delta_y*i
    
    # definition of the y-dependent pressure function
    p = p_a + p_b * np.cos(y*np.pi/L)
    
    # initialisation of the numerical gradient of pressure
    p_dash = np.zeros(N+1)
    
    
    # first point value for the pressure gradient is obtained through 1st order
    # forward difference formula
    p_dash[0] = (p[1] - p[0])/Delta_y
    
    # values for the points with index between 1 and N-1 are obtained through 
    # 2nd order finite differences formula, calculated by utilising a for loop
    for i in range(1,N):
        p_dash[i] = (p[i+1] - p[i-1])/(2*Delta_y)
    
    # last point value is obtained through 1st order backward difference formula
    p_dash[N] = (p[N] - p[N-1])/Delta_y
    
    # multiplying p_dash by appropriate constants speed u is obtained
    # (the n stands for numerical solution)
    u_n = -(1 /(rho*f))*p_dash
    
    # defining the analytical solution to be compared with the numerical one
    # (a stands for analytical solution)
    u_a = (p_b*np.pi*np.sin(np.pi*y/L))/(rho*f*L)
    
    # returning two arrays: one for the positions and two for the speed
    # (numerical and analitycal)
    return y, u_n, u_a
    
def order_accuracy(num=6, rho=1.0, p_a=1e5, p_b=200.0, f=1e-4, L=2.4e6, \
                   y_min=0.0, y_max=1e6):
    
    num=int(num)   # making sure that num is an integer
    
    # initialising array er for storing error values and relative step width
    er=np.zeros((num,2))
    
    # evaluating errors for different step widths, relatively to the point L/2
    # which is located in the N/2-th position of array u_n, containing numerical
    # solution
    for i in range(1, num+1):
        N=10**i     # the step width increases 10 times each iteration
        y, u_n, u_a = geostrophic_wind(rho, p_a, p_b, f, L, y_min, y_max, N)
        er[i-1,0] = (y_max - y_min)/N
        er[i-1,1] = abs(u_n[int(N/2)] - u_a[int(N/2)])
    
    # initialising array n for storing order of convergence n values
    n = np.zeros(num-1)
    
    # order of convergence values are calculated using each couple of
    #consecutive points from array er
    for i in range(num-1):
        n[i] = (np.log(er[i+1,1]) - np.log(er[i,1]))/\
                (np.log(er[i+1,0]) - np.log(er[i,0]))
    
    # definition of arrays to be used in plotting the points obtained for n
    step = er[:,0]
    epsilon = er[:,1]
    plt.loglog(step,epsilon)
    
    # finally, values of n are returned for analysis
    return n

def geowind_accurate(rho=1.0, p_a=1e5, p_b=200.0, f=1e-4, L=2.4e6, y_min=0.0, \
                     y_max=1e6, N=1e5):
    # conversion of N to an integer, if int(N) != N then TypeError is raised
    N = int(N)
    
    # initialisation of the y array
    y = np.zeros(N+1)
    Delta_y=(y_max - y_min)/N
    for i in range(N+1):
        y[i] = y_min + Delta_y*i
    
    # definition of the y-dependent pressure function
    p = p_a + p_b * np.cos(y*np.pi/L)
    
    # initialisation of the numerical gradient of pressure
    p_dash = np.zeros(N+1)
    
    
    # values for the points with index between 1 and N-1 are obtained through 
    # 2nd order finite differences formula, calculated by utilising a for loop
    for i in range(0,N-5):
        p_dash[i] = (9*p[i+4] - 16*p[i+3] - 36*p[i+2] + 144*p[i+1] - 101*p[i])/(60*Delta_y)
    
    # last points values are obtained 
    for i in range(N-5,N):
        p_dash[i] = (p[i+1] - p[i-1])/(2*Delta_y)
        
    p_dash[N] = (p[N] - p[N-1])/Delta_y
    
    # multiplying p_dash by appropriate constants speed u is obtained
    # (the n stands for numerical solution)
    u_n = -(1 /(rho*f))*p_dash
    
    # defining the analytical solution to be compared with the numerical one
    # (a stands for analytical solution)
    u_a = (p_b*np.pi*np.sin(np.pi*y/L))/(rho*f*L)
    
    # returning two arrays: one for the positions and two for the speed
    # (numerical and analitycal)
    return y, u_n, u_a
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    