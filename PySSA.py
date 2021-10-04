# This code is a Python implementation of the Synchrotron Self Absorbed model
# as described in Soderberg et al. 2005 (https://ui.adsabs.harvard.edu/abs/2005ApJ...621..908S/abstract) 

# Import required packages
import numpy as np
from scipy import integrate,special
import astropy.constants as const
import astropy.units as u

#--------------------------------------------------------------

## Main physical variables 
# distance to source and explosion time

d = (23.0 * u.Mpc).to(u.cm)       # cm ; distance to source
t_exp = 2453216.7                 # JD ; time of explosion
t_0 = t_exp + 10                  # reference time 10 days since explosion
eta = 4                           # shell radius to thickness factor 

# B_0, r_0, alpha_r, p, nu_m_0, s, t_exp, zeta

B_0 = 1.06 * u.G       # G
r_0 = 5.0e15 * u.cm    # cm
alpha_r = 0.9  
p = 3.0
nu_m_0 = 0.02e9 * u.Hz # Hz 
s = 2.0
zeta = 1.0 

scriptF_0 = 1.0        # as we have eps_e = eps_B

# Physical constants

m_e = const.m_e.cgs    # g
e = const.e.esu        # esu
c = const.c.cgs        # cm/s

#--------------------------------------------------------------

# Modified Bessel Function of second kind  

def K_n(x,n):
    """
    Modified Bessel Function formulae from Wolfram (see https://mathworld.wolfram.com/ModifiedBesselFunctionoftheSecondKind.html)

    Returns value of K_n of the order of n at values of x
    """
    fy = lambda y: np.exp(-x*y) * ((y**2.0 - 1.0)**(n-0.5)) 
    K_n = (np.sqrt(np.pi)/special.factorial(n - 0.5)) * ((0.5*x)**n) * integrate.quad(fy,1.0,np.inf)[0]
    #K_n_error = (np.sqrt(np.pi)/special.factorial(n - 0.5)) * ((0.5*)**n) * integrate.quad(fy,1.0,np.inf)[1]
    return K_n

#--------------------------------------------------------------

# Define functions F, F_2 and F_3

def F(x):
    """
    Returns value of function F defined in equation A7 of Soderberg et al. 2005
    """
    F_x = []
    for x_i in x:
        fy = lambda t: K_n(y,5.0/3.0)
        F_x.append(x_i * integrate.quad(fy,x_i,np.inf)[0])
    return np.array(F_x)

def F_2(x,F,p=p):
    F_2_x = []
    for xi in x:
        fy = lambda y: F(y) * (y**(p-2.0)/2.0)
        F_2_x.append(np.sqrt(3) * integrate.quad(fy,0,x_i)[0])
    return np.array(F_2_x)

def F_3(x,F,p=p):
    F_3_x = []
    for xi in x:
        fy = lambda y: F(y) * (y**(p-3.0)/2.0)
        F_3_x.append(np.sqrt(3) * integrate.quad(fy,0,x_i)[0])
    return np.array(F_3_x)

#--------------------------------------------------------------

# Define constants C_f and C_tau in terms of the physical constants above

gamma_m_0 = (np.sqrt(2.0*np.pi*nu_m_0*(m_e*c/(e*B_0)))).value

C_f = ((2.0*np.pi/(2.0+p)) * m_e * ((r_0/d)**2.0) * ((2.0*np.pi*m_e*c/(e*B_0))**0.5)).value

C_tau = (((p+2.0)*(p-2.0)/(4.0*np.pi*eta)) * (gamma_m_0**(p-1.0)) \
        * ((B_0**2.0)/(8.0*np.pi)) * scriptF_0 \
        * (((e**3.0)*B_0*r_0)/((m_e**3.0)*(c**4.0)*gamma_m_0)) \
        * (((e*B_0)/(2.0*np.pi*m_e*c))**(p/2.0))).value \

