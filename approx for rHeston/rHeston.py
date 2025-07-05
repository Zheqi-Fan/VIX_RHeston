import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from scipy.special import gamma
from scipy.integrate import quad, trapz
import pandas as pd


# rough Heston class
class roughHeston:
    
    def __init__(self, nbTimeSteps, heston_params, T):
        # Time discretisation parameters
        self.T = T
        self.n = nbTimeSteps
        self.dt = self.T / self.n
        self.time_grid = np.linspace(0., T, self.n + 1)

        # Heston model paramters
        self.S0 = heston_params['S0']
        self.kappa = heston_params['kappa']
        self.nu = heston_params['nu'] 
        self.theta = heston_params['theta']
        self.alpha = heston_params['alpha']
        self.V0 = heston_params['V0']
        self.rho = heston_params['rho']

        # Precomputations to speed up pricing
        self.frac = self.dt**self.alpha / gamma(self.alpha + 2.)
        self.frac2 = self.dt**self.alpha / gamma(self.alpha + 1.)
        self.frac_bar = 1. / gamma(1.-self.alpha)
        self.fill_a()
        self.fill_b()

    # Fractional Riccati equation
    def F(self, a, x):
        '''
        Euch2019 MF
        Sec 5.1, eq 5.1
        '''
        return -0.5*(a*a + 1j *a) - (self.kappa - 1j*a*self.rho*self.nu)*x + 0.5*self.nu*self.nu*x*x

    # Filling the coefficient a and b which don't depend on the characteristic function
    def a(self, j, k):
        if j == 0:
            res = ((k - 1)**(self.alpha + 1) - (k - self.alpha - 1)*k**self.alpha)
        elif j == k:
            res = 1.
        else:
            res = ((k + 1 - j)**(self.alpha + 1) + (k - 1 - j)**(self.alpha + 1) - 2 * (k - j)**(self.alpha + 1))

        return self.frac*res

    def fill_a(self):
        self.a_ = np.zeros(shape = (self.n + 1, self.n + 1))
        for k in range(1, self.n + 1):
            for j in range(k + 1):
                self.a_[j, k] = self.a(j, k)

    def b(self, j, k):
        return self.frac2*((k - j)**self.alpha - (k - j - 1)**self.alpha)

    def fill_b(self):
        self.b_ = np.zeros(shape = (self.n, self.n + 1))
        for k in range(1, self.n + 1):
            for j in range(k):
                self.b_[j, k] = self.b(j, k)

    # Computation of two sums used in the scheme
    def h_P(self, a, k):
        res = 0
        for j in range(k):
            res += self.b_[j, k] * self.F(a, self.h_hat[j])
        return res

    def sum_a(self, a, k):
        res = 0
        for j in range(k):
            res += self.a_[j, k] * self.F(a, self.h_hat[j])
        return res

    # Solving function h for each time step
    def fill_h(self, a):
        self.h_hat = np.zeros((self.n + 1), dtype=complex)
        for k in range(1, self.n + 1):
            h_P = self.h_P(a, k)
            sum_a = self.sum_a(a, k)
            self.h_hat[k] = sum_a + self.a_[k, k]*self.F(a, h_P)

    # Characteristic function computation
    def rHeston_char_function(self, a):
        '''
        Euch2019 MF
        eq 4.5
        integral: I^1h(a,t)
        frac_integral: I^(1-alpha)h(a,t)
        '''
        # Filling the h function
        self.fill_h(a)

        # Standard integral of the h function
        integral = trapz(self.h_hat, self.time_grid)

        # Fractional integral of the h function
        func = lambda s: (self.T - s)**(1. - self.alpha)
        y = np.fromiter((((func(self.time_grid[i]) - func(self.time_grid[i+1]))*self.h_hat[i]) for i in range(self.n)), self.h_hat.dtype)
        frac_integral = self.frac_bar * np.sum(y) / (1.-self.alpha)

        # Characteristic function
        return np.exp(self.kappa*self.theta*integral + self.V0*frac_integral)

    # Pricing with an inverse Fourier transform
    def rHeston_Call(self, k, upLim):
        '''
        Lewis method
        see Yves Page108 (Chinese version)
        '''
        K = self.S0*np.exp(k)
        func = lambda u: np.real(np.exp(-1j*u*k)*self.rHeston_char_function(u-0.5*1j)) / (u**2 + 0.25)
        # integ = quad(func, 0, 5.)
        integ = quad(func, 0, upLim)
        return self.S0 - np.sqrt(self.S0*K) * integ[0] / np.pi

    # Analytical formula for the standard Heston characteristic function
    def heston_char_function(self,u):
        nu2 = self.nu**2
        T = self.T
        dif = self.kappa - self.rho*self.nu*u*1j
        d = np.sqrt(dif**2 + nu2 *(1j*u + u**2))
        g = (dif - d) / (dif + d)
        return np.exp(1j*u*(np.log(self.S0)))\
               *np.exp((self.kappa*self.theta/nu2) * ((dif-d)*T - 2.*np.log((1. - g*np.exp(-d*T))/(1.-g))))\
               *np.exp((self.V0/nu2) * (dif-d)*(1.-np.exp(-d*T))/(1-g*np.exp(-d*T)))

    # Pricing with an inverse Fourier transform
    def heston_Call(self, k):
        K = self.S0 * np.exp(k)
        func = lambda u: np.real(np.exp(-1j*u*k) * self.heston_char_function(u-0.5*1j)) / (u**2+0.25)
        integ = quad(func, 0, np.inf)
        return self.S0 - np.sqrt(self.S0*K) * integ[0] / np.pi