# This code is a Python implementation of the Synchrotron Self Absorbed model
# as described in Soderberg et al. 2005 (https://ui.adsabs.harvard.edu/abs/2005ApJ...621..908S/abstract)

## Import required packages ##

import numpy as np
from scipy import integrate, special
import astropy.constants as const
import astropy.units as u
import pickle
from scipy import interpolate

## Physical Constants ##

m_e = (const.m_e.cgs).value  # g
e = (const.e.esu).value  # esu
c = (const.c.cgs).value  # cm/s

##--------------------------------------------------------------

## Functions ##

# --------------------------------------------------------------


# Define functions F_2 and F_3

to_interp = False

# for function F2

F2_file = open("Interpolation_files/F2_values.pkl", "rb")
F2_dict = pickle.load(F2_file)
F2_x_values = F2_dict['x']
F2_p_values = F2_dict['p']
F2_values = F2_dict['F2']

F2_grid = np.array(F2_values).reshape(len(F2_p_values), len(F2_x_values))



# for function F3

F3_file = open("Interpolation_files/F3_values.pkl", "rb")
F3_dict = pickle.load(F3_file)
F3_x_values = F3_dict['x']
F3_p_values = F3_dict['p']
F3_values = F3_dict['F3']

F3_grid = np.array(F3_values).reshape(len(F3_p_values), len(F3_x_values))




def calc_F(x):
    """
    Returns values of function F (defined in eq. A7 of Soderberg et al. 2005)
    at x
    """

    def fy1(y):
        return special.kv(5.0 / 3.0, y)

    if isinstance(x, float):
        return x * integrate.quad(fy1, x, np.inf)[0]
    else:
        F_x = np.zeros(len(x))
        for i, x_i in enumerate(x):
            F_x[i] = x_i * integrate.quad(fy1, x_i, np.inf)[0]
        return F_x

    
    
def calc_F_2(x, calc_F, p):
    """
    Returns values of function F2 (defined in eq. A7 of Soderberg et al. 2005)
    at x
    """

    def fy2(y):
        return calc_F(y) * (y ** ((p - 2.0) / 2.0))

    if isinstance(x, float):
        return np.sqrt(3) * integrate.quad(fy2, 0, x)[0]
    else:
        F_2_x = np.zeros(len(x))
        for i, x_i in enumerate(x):
            if x_i < 20000:
                F_2_x[i] = np.sqrt(3) * integrate.quad(fy2, 0, x_i)[0]
            else:
                F_2_x[i:] = np.sqrt(3) * integrate.quad(fy2, 0, 20000)[0]
        return F_2_x

def calc_F_2_interp(x,p):    
    
    interp_func_F2 = interpolate.RegularGridInterpolator((np.unique(F2_x_values), np.unique(F2_p_values)), F2_grid.T, bounds_error = False, method="linear", fill_value=None)
    return interp_func_F2((x,p))
        
        
def calc_F_3(x, calc_F, p):
    """
    Returns values of function F3 (defined in eq. A7 of Soderberg et al. 2005)
    at x
    """

    def fy3(y):
        return calc_F(y) * (y ** ((p - 3.0) / 2.0))

    if isinstance(x, float):
        return np.sqrt(3) * integrate.quad(fy3, 0, x)[0]
    else:
        F_3_x = np.zeros(len(x))
        for i, x_i in enumerate(x):
            if x_i < 2000:
                F_3_x[i] = np.sqrt(3) * integrate.quad(fy3, 0, x_i)[0]
            else:
                F_3_x[i:] = np.sqrt(3) * integrate.quad(fy3, 0, 2000)[0]
        return F_3_x

def calc_F_3_interp(x,p):  

    interp_func_F3 = interpolate.RegularGridInterpolator((np.unique(F3_x_values), np.unique(F3_p_values)), F3_grid.T, bounds_error = False, method="linear", fill_value=None)
    return interp_func_F3((x,p))

# --------------------------------------------------------------

# Define constants gamma_m_0, C_f, C_tau, alpha_gamma, alpha_B in terms of the physical variables/constants above


def calc_gamma_m_0(B_0, nu_m_0):
    """
    Calculate gamma_m_0 from eq. A15 of Soderberg et al. 2005
    """
    return np.sqrt(2.0 * np.pi * nu_m_0 * (m_e * c / (e * B_0)))


def calc_C_f(B_0, r_0, d, p):
    """
    Calculate constant C_f using eq. A13 of Soderberg et al. 2005
    """
    return (
        (2.0 * np.pi / (2.0 + p))
        * m_e
        * ((r_0 / d) ** 2.0)
        * ((2.0 * np.pi * m_e * c / (e * B_0)) ** 0.5)
    )


def calc_C_tau(B_0, r_0, eta, gamma_m_0, p, scriptF_0):
    """
    Calculate constant C_tau using eq. A14 of of Soderberg et al. 2005
    """
    eq1 = ((p + 2.0) * (p - 2.0) / (4.0 * np.pi * eta)) * (gamma_m_0 ** (p - 1.0))
    eq2 = ((B_0**2.0) / (8.0 * np.pi)) * scriptF_0
    eq3 = ((e**3.0) * B_0 * r_0) / ((m_e**3.0) * (c**4.0) * gamma_m_0)
    eq4 = ((e * B_0) / (2.0 * np.pi * m_e * c)) ** (p / 2.0)
    return eq1 * eq2 * eq3 * eq4


def calc_alpha_gamma(alpha_r):
    """
    Calculate constant alpha_gamma using eq. 9 of Soderberg et al. 2005
    """
    return 2.0 * (alpha_r - 1.0)


def calc_alpha_B(alpha_r, s):
    """
    Calculate constant alpha_gamma using eq. 10 of Soderberg et al. 2005
    """
    return ((2.0 - s) / 2.0) * alpha_r - 1.0


# --------------------------------------------------------------

# Calculate the characteristic frequencies as a function of time


def calc_nu_m(t, nu_m_0, t_0, alpha_gamma, alpha_B):
    """
    Calculate nu_m at t from above calcuated constants (see eq. A10 of Soderberg et al. 2005)
    """
    return nu_m_0 * (t / t_0) ** (2.0 * alpha_gamma + alpha_B)


# --------------------------------------------------------------

# Caluclate the optical depth tau_nu at a given time


def calc_tau_nu(t, t_0, C_tau, alpha_r, alpha_gamma, alpha_B, alpha_scrpitF, p, nu, F2):
    """
    Calculate tau_nu at t (see eq. A9 of Soderberg et al. 2005)
    """
    return (
        C_tau
        * (
            (t / t_0)
            ** (
                (p - 2.0) * alpha_gamma
                + (3.0 + p / 2.0) * alpha_B
                + alpha_r
                + alpha_scrpitF
            )
        )
        * (nu ** (-(p + 4.0) / 2.0))
        * F2
    )


# --------------------------------------------------------------

# Finally calculate flux density


def calc_f_nu(t, t_0, C_f, alpha_r, alpha_B, tau_nu, xi, p, nu, F2, F3):
    """
    Calculate f_nu at t (see eq. A8 of Soderberg et al. 2005)
    """
    eq1 = C_f * ((t / t_0) ** ((4.0 * alpha_r - alpha_B) / 2.0))
    eq2 = (
        ((1.0 - np.exp(-(tau_nu ** (xi)))) ** (1.0 / xi))
        * (nu ** (5.0 / 2.0))
        * F3
        / F2
    )
    return (((eq1 * eq2) * (u.erg / u.s / (u.cm) ** 2.0 / u.Hz)).to(u.uJy)).value


# --------------------------------------------------------------

# One function to calculate the entire thing


def SSA_flux_density(
    t, t_0, nu, d, eta, B_0, r_0, alpha_r, p, nu_m_0, s, xi, scriptF_0, alpha_scriptF
, to_interp):
    """
    Returns SSA flux density (in mJy) as a function of time
    """
    log_r_0 = np.log10(r_0)
    log_nu_m_0 = np.log10(nu_m_0)

    alpha_gamma = calc_alpha_gamma(alpha_r)
    alpha_B = calc_alpha_B(alpha_r, s)
    gamma_m_0 = calc_gamma_m_0(B_0, 10 ** (log_nu_m_0))
    C_tau = calc_C_tau(B_0, 10 ** (log_r_0), eta, gamma_m_0, p, scriptF_0)
    C_f = calc_C_f(B_0, 10 ** (log_r_0), d, p)

    nu_m = calc_nu_m(t, 10 ** (log_nu_m_0), t_0, alpha_gamma, alpha_B)
    x = (2.0 / 3.0) * (nu / nu_m)
    print(x)
    if to_interp==False:
        F2 = calc_F_2(x, calc_F, p)
        F3 = calc_F_3(x, calc_F, p)
    else:
        F2 = calc_F_2_interp(x, p)
        F3 = calc_F_3_interp(x, p)        

    tau_nu = calc_tau_nu(
        t, t_0, C_tau, alpha_r, alpha_gamma, alpha_B, alpha_scriptF, p, nu, F2
    )
    f_nu = calc_f_nu(t, t_0, C_f, alpha_r, alpha_B, tau_nu, xi, p, nu, F2, F3)

    return f_nu


###--------------------------------------------------------------

### Test Calculations ##

# import time
# from astropy.table import Table
#
####--------------------------------------------------------------
#
#### Use the following lines of code to generate an SSA lightcurve with your own
#### set of parameters.
###--------------------------------------------------------------
##
#### Main physical parameters
#
# d = ((23.0 * u.Mpc).to(u.cm)).value  # cm ; distance to source
# t_exp = 2453216.7  # JD ; time of explosion
# t_0 = 10  # reference time 10 days since explosion
# eta = 4  # shell radius to thickness factor
# nu = 3.0e09  # * u.Hz                      # frequency of observation in Hz
#
#
# B_0 = 1.06  # * u.G                  # G
# r_0 = 5.0e15  # * u.cm               # cm
# alpha_r = 0.9
# p = 3.0
# nu_m_0 = 0.02e9  # * u.Hz            # Hz
# s = 2.0
# xi = 1.0
#
# scriptF_0 = 1.0  # as we have eps_e = eps_B
# alpha_scriptF = 0.0  # as we have eps_e = eps_B at all times
#
## Physical constants
#
# m_e = (const.m_e.cgs).value  # g
# e = (const.e.esu).value  # esu
# c = (const.c.cgs).value  # cm/s
#
#
## --------------------------------------------------------------
#
# tstart = time.time()
#
# t = np.arange(1, 100)
#
## f_nu = SSA_flux_density(t,t_0,nu,d,eta,B_0,r_0,alpha_r,p,nu_m_0,s,xi,scriptF_0,alpha_scriptF)
#
## data_tab = Table([t,f_nu], names=['time', 'flux_density'])
#
## data_tab.write('SSA_flux_density_test.txt', format='ascii')
#
# t = Table.read("soderberg_data.tex", format="latex")
#
# print(t)

############
##
