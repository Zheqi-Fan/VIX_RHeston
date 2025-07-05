#IMPORTING THE LIBRARIES
import random as rd
import numpy as np 
import matplotlib.pyplot as plt 
import statistics as stat
import math
from math import *
import pandas as pd
from scipy import integrate as intg
from scipy.stats import linregress
import warnings
import scipy.stats as si
np.seterr(divide='ignore', invalid='ignore')
import scipy.integrate as integrate
from scipy.special import gamma
import cmath
import scipy
warnings.filterwarnings('ignore')

#function that creates the vector c according to the number of factor n and the value of rn
def vec_c(n,rn,alpha):
    # c=[0]*n
    c = np.zeros(n)
    for i in range(n):
        c[i]=((np.power(rn,1-alpha)-1)/(gamma(alpha)*gamma(2-alpha)))*np.power(rn,(1-alpha)*(i+1-1-n/2))
    return(c)

#function that creates the vector x according to the number of factor n and the value of rn
def vec_x(n,rn,alpha):
    # x=[0]*n
    x = np.zeros(n)
    for i in range(n):
        x[i]=(((1-alpha)*(np.power(rn,2-alpha)-1))/((2-alpha)*(np.power(rn,1-alpha)-1)))*np.power(rn,i+1-1-n/2)
    return(x)

def g0(u, c, x, V0,lamb,theta,nu,rho):
    '''
    Eq(18), function
    c: array, coefficients
    x: array, mean-reverting speed
    '''
    a = 0
    for i in range(len(c)):
        a = a + c[i]*(1-exp(-x[i]*u))/x[i]
    res = V0+lamb*theta*a
    return(res)

###### first part
def integ_g0(t, c, x, V0,lamb,theta,nu,rho, tau):
    '''
    Eq(27), constant 
    in VIX formulation
    t: current time
    tau: time duration
    '''
    c, x = np.array(c), np.array(x)
    
    vec1 = c/x
    vec2 = tau + (np.exp(-x*(t+tau)) - np.exp(-x*t))/x
    return V0*tau + lamb*theta*np.dot(vec1, vec2)


###### basic building blocks
def coef_matrix(t, c, x, V0,Ut,lamb,theta,nu,rho, tau):
    '''
    Eq(11), constant
    D-lambda*1*c
    '''
    assert len(c) == len(x), "x and c should have same length"
    c, x = np.array(c), np.array(x)
    
    D = np.diag(-np.array(x))
    return D - lamb*np.ones(len(x)).reshape(-1,1) @ np.array(c).reshape(1,-1)
def get_M(t, c, x, V0,Ut,lamb,theta,nu,rho, tau):
    '''
    Eq(12), constant
    the column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i]

    See https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
    '''
    c, x = np.array(c), np.array(x)
    
    cof_mat = coef_matrix(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
    eigenvalues, eigenvectors = np.linalg.eig(cof_mat)
    return eigenvectors
def get_omega(t, c, x, V0,Ut,lamb,theta,nu,rho, tau):
    '''
    Eq(11), constant
    '''
    c, x = np.array(c), np.array(x)
    
    cof_mat = coef_matrix(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
    eigenvalues, eigenvectors = np.linalg.eig(cof_mat)
    return np.abs(eigenvalues)
def get_E(t, c, x, V0,Ut,lamb,theta,nu,rho, tau, s):
    '''
    Eq(12), function
    s: intermediate time
    '''
    c, x = np.array(c), np.array(x)
    
    cof_mat = coef_matrix(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
    eigenvalues, eigenvectors = np.linalg.eig(cof_mat)
    omega = get_omega(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
    if((omega+eigenvalues).any()): 
        # if exist non-zeros, then report messages
        print('check eigens')
    array_diag = np.exp(-omega*s)
    return np.diag(array_diag)


###### advanced blocks
def get_Hh(t, c, x, V0,Ut,lamb,theta,nu,rho, tau):
    '''
    Eq(29), function
    '''
    omega = get_omega(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
    Hh_diag = (np.exp(-omega*t) - np.exp(-omega*(t+tau)))/omega
    return np.diag(Hh_diag)
def get_Hp(t, c, x, V0,Ut,lamb,theta,nu,rho, tau):
    '''
    Eq(32), constant
    '''
    c, x  = np.array(c), np.array(x)
    nDim  = len(c)
    omega = get_omega(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
    array_diag = np.zeros(nDim)
    for k in range(nDim):
        #### 1st term
        temp1 = lamb * theta * np.dot(np.ones(nDim), c/x)
        temp2 = tau + 1/omega[k] * (np.exp(-omega[k]*tau)-1)
        term1 = 1/omega[k] * (V0+temp1) * temp2

        #### 2nd term
        temp3 = c/(x*(omega[k]-x))
        temp4 = -1/x * (np.exp(-x*(t+tau)) - np.exp(-x*t))
#         temp5 = 1/omega[k] * np.exp(-(x-omega[k])*t) * (np.exp(-omega[k]*(t+tau)) - np.exp(-omega[k]*t))
        temp5 = 1/omega[k] * np.exp(-x*t) * (np.exp(-omega[k]*tau) - 1)
        term2 = -lamb*theta * np.dot(np.ones(nDim), temp3*(temp4+temp5))
        
        array_diag[k] = term1 + term2
    return np.diag(array_diag)

# def get_a(t, c, x, V0,Ut,lamb,theta,nu,rho, tau):
#     '''
#     Eq(23), constant, 可能需要调整
#     '''
#     c, x = np.array(c), np.array(x)

#     ## compute E_inv
#     omega = get_omega(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
#     array_diag = np.exp(omega*t)
#     E_inv = np.diag(array_diag)

#     ## compute M_inv
#     M = get_M(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
#     M_inv = np.linalg.inv(M)
    
#     # F_t = get_F_initial_condition, zero-vector, 考虑加入U_t的调整, 现在默认t=0
# #     F_t = np.zeros(len(c)).reshape([-1,1])
#     F_t = Ut.reshape([len(c),-1])
#     return E_inv @ M_inv @ F_t

# def get_Fh(t, c, x, V0,Ut,lamb,theta,nu,rho, tau, s):
#     '''
#     Eq(14), function
#     '''
#     M = get_M(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
#     E_s = get_E(t, c, x, V0,Ut,lamb,theta,nu,rho, tau, s)
#     a = get_a(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
#     return M @ E_s @ a

def get_Fh(t, c, x, V0,Ut,lamb,theta,nu,rho, tau, s):
    '''
    Eq(14), function, without using get_a()
    '''
    M = get_M(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
    Es_by_Et = np.diag(np.exp(x*(t-s)))
    M_inv = np.linalg.inv(M)
    return M @ Es_by_Et @ M_inv @ Ut.reshape([len(c),-1])


def get_G(t, c, x, V0,Ut,lamb,theta,nu,rho, tau, s):
    '''
    Eq(19), function
    used in calculation of Fp(s)
    '''
    c, x  = np.array(c), np.array(x)
    nDim  = len(c)
    omega = get_omega(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
    array_G_tilda = np.zeros(nDim)
    for i in range(nDim):
        #### 1st term
        temp1 = lamb * theta * np.dot(np.ones(nDim), c/x)
        term1 = 1/omega[i] * (V0+temp1) * (np.exp(omega[i]*s) - np.exp(omega[i]*t))
        #### 2nd term
        temp2 = c/(x*(omega[i]-x)) * (np.exp((omega[i]-x)*s) - np.exp((omega[i]-x)*t))
        temp3 = np.dot(np.ones(nDim), temp2)
        term2 = -lamb*theta*temp3
        #### G_tilda
        array_G_tilda[i] = term1 + term2
    return np.diag(array_G_tilda)
def get_Fp(t, c, x, V0,Ut,lamb,theta,nu,rho, tau, s):
    '''
    Eq(18), function
    '''
    M   = get_M(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
    E_s = get_E(t, c, x, V0,Ut,lamb,theta,nu,rho, tau, s)
    X_s = M @ E_s
    G_s = get_G(t, c, x, V0,Ut,lamb,theta,nu,rho, tau, s)
    m   = np.linalg.inv(M) @ np.ones(len(c)).reshape(-1,1)
    return -lamb * X_s @ G_s @ m
def get_F(t, c, x, V0,Ut,lamb,theta,nu,rho, tau, s):
    '''
    Eq(20), function
    '''
    Fh = get_Fh(t, c, x, V0,Ut,lamb,theta,nu,rho, tau, s)
    Fp = get_Fp(t, c, x, V0,Ut,lamb,theta,nu,rho, tau, s)
    return Fh+Fp


def integ_F(t, c, x, V0,Ut,lamb,theta,nu,rho, tau):
    '''
    Eq(28),
    in VIX formulation
    '''
    c, x = np.array(c), np.array(x)
    nDim = len(c)
    
    M = get_M(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
    M_inv = np.linalg.inv(M)
    m = np.linalg.inv(M) @ np.ones(nDim).reshape(-1,1)
    Hp = get_Hp(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
#     Hh = get_Hh(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
#     a = get_a(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
#     term1 = M @ Hh @ a
    F_t = Ut.reshape([len(c),-1])
    array_temp = (1-np.exp(-x*tau))/x
    diag_mat = np.diag(array_temp)
    term1 = M @ diag_mat @ M_inv @ F_t
    term2 = lamb * M @ Hp @ m
    return term1 - term2


#### single-factor VIX-square
def squared_VIX(t, c, x, V0,Ut,lamb,theta,nu,rho, tau):
    c, x = np.array(c), np.array(x)
    term1 = integ_g0(t, c, x, V0,lamb,theta,nu,rho, tau)
    term2 = c.reshape([1,-1]) @ integ_F(t, c, x, V0,Ut,lamb,theta,nu,rho, tau)
    #### calculate Eq(25)
    integ_EV = term1 + term2

    VIX2 = 1/tau * integ_EV * 100**2
    return VIX2[0]


def K0(t, c, x, V0,Ut,lamb,theta,nu,rho, tau):
    
    pass

