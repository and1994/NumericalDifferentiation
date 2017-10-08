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
    if not(isinstance(float(p_b),float) and float(p_b) > 0):   # >0 is it necessary?
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
    if not(callable(f)):
        raise Exception('Error in geostrophic_wind: A callable function must be sent\
        to integrate')
    
    # conversion of N to an integer, if int(N) != N then TypeError is raised
    N=int(N)
    
    # initialisation of the y array
    y = np.linspace(y_min, y_max, N+1)
    
    # definition of the y-dependent pressure function
    p = p_a + p_b * np.cos(y*np.pi/L)
    
    # initialisation of the numerical gradient of pressure
    p_dash = np.zeros(N+1)
    
    # definition of the step used in following integration, i.e. distance
    # between consecutive y values, which are all the same because np.linspace
    # is utilised in y initialisation
    Delta_y=y[1]-y[0]
    
    # first point value for the pressure gradient is obtained through 1st order
    # forward difference formula
    p_dash[0]=(p[1]-p[0])/(Delta_y)
    
    # values for the points with index between 1 and N-1 are obtained through 
    # 2nd order finite differences formula, calculated by utilising a for loop
    for i in range(1,N):
        p_dash[i]=(p[i+1]-p[i-1])/(2*Delta_y)
    
    # last point value is obtained through 1st order backward difference formula
    p_dash[N]=(p[N]-p[N-1])/(Delta_y)
    
    # finally, multiplying p_dash by appropriate constants speed u is obtained 
    u = -(1/(rho*f))*p_dash