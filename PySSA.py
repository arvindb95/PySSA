# This code is a Python implementation of the Synchrotron Self Absorbed model
# as described in Soderberg et al. 2005 (https://ui.adsabs.harvard.edu/abs/2005ApJ...621..908S/abstract) 

# Import required packages
import numpy as np
from scipy import integrate,special
import astropy.constants as const
import astropy.units as u
import time
from astropy.table import Table
import matplotlib.pyplot as plt

from alive_progress import alive_bar

## Functions ##

#--------------------------------------------------------------

# Define functions F, F_2 and F_3

def calc_F(x):
    """
    Returns values of function F (defined in eq. A7 of Soderberg et al. 2005)
    at x
    """
    if isinstance(x,float):
        fy = lambda y: special.kv(5.0/3.0,y)
        return (x * integrate.quad(fy,x,np.inf)[0])
    else:
        F_x = np.zeros(len(x))
        for i,x_i in enumerate(x):
            fy = lambda y: special.kv(5.0/3.0,y)
            F_x[i] = (x_i * integrate.quad(fy,x_i,np.inf)[0])
        return F_x
    
def calc_F_2(x,calc_F,p):
    """
    Returns values of function F2 (defined in eq. A7 of Soderberg et al. 2005) 
    at x
    """
    if isinstance(x,float):
        fy = lambda y: calc_F(y) * (y**((p-2.0)/2.0))
        return np.sqrt(3) * integrate.quad(fy,0,x)[0]
    else:    
        F_2_x = np.zeros(len(x))
        print("Calculating function F2....")
        with alive_bar(len(x)) as bar:
            for i,x_i in enumerate(x):
                #counter += 1
                #print("------------------------------------------------------")
                #print("calculating F2 for iteration number : ",counter," out of ",len(x))
                fy = lambda y: calc_F(y) * (y**((p-2.0)/2.0))
                F_2_x[i] = (np.sqrt(3) * integrate.quad(fy,0,x_i)[0])
                bar()
        sel_large_x = np.where(x >= 22778.5451769)
        F_2_x[sel_large_x] = F_2_x[sel_large_x][0]    
        print("===========================")
        return F_2_x

def calc_F_3(x,calc_F,p):
    """
    Returns values of function F3 (defined in eq. A7 of Soderberg et al. 2005) 
    at x
    """
    if isinstance(x,float):
        fy = lambda y: calc_F(y) * (y**((p-3.0)/2.0))
        return np.sqrt(3) * integrate.quad(fy,0,x)[0]
    else:
        F_3_x = np.zeros(len(x))
        print("Calculating function F3....")
        with alive_bar(len(x)) as bar:
            for i,x_i in enumerate(x):
                #counter += 1
                #print("------------------------------------------------------")
                #print("calculating F3 for iteration number : ",counter," out of ",len(x))
                fy = lambda y: calc_F(y) * (y**((p-3.0)/2.0))
                F_3_x[i] = (np.sqrt(3) * integrate.quad(fy,0,x_i)[0])
                bar()
        sel_large_x = np.where(x >= 22778.5451769)
        F_3_x[sel_large_x] = F_3_x[sel_large_x][0]
        print("===========================")
        return F_3_x

#--------------------------------------------------------------

# Define constants gamma_m_0, C_f, C_tau, alpha_gamma, alpha_B in terms of the physical variables/constants above

def calc_gamma_m_0(B_0,nu_m_0):
    """
    Calculate gamma_m_0 from eq. A15 of Soderberg et al. 2005
    """
    return (np.sqrt(2.0*np.pi*nu_m_0*(m_e*c/(e*B_0)))).value

def calc_C_f(B_0,r_0,d,p):
    """
    Calculate constant C_f using eq. A13 of Soderberg et al. 2005
    """
    return ((2.0*np.pi/(2.0+p)) * m_e * ((r_0/d)**2.0) * ((2.0*np.pi*m_e*c/(e*B_0))**0.5)).value

def calc_C_tau(B_0,r_0,eta,gamma_m_0):
    """
    Calculate constant C_tau using eq. A14 of of Soderberg et al. 2005 
    """
    eq1 = ((p+2.0)*(p-2.0)/(4.0*np.pi*eta)) * (gamma_m_0**(p-1.0)) 
    eq2 = ((B_0**2.0)/(8.0*np.pi)) * scriptF_0 
    eq3 = (((e**3.0)*B_0*r_0)/((m_e**3.0)*(c**4.0)*gamma_m_0)) 
    eq4 = (((e*B_0)/(2.0*np.pi*m_e*c))**(p/2.0)) 
    return (eq1 * eq2 * eq3 *eq4).value

def calc_alpha_gamma(alpha_r):
    """
    Calculate constant alpha_gamma using eq. 9 of Soderberg et al. 2005
    """
    return 2.0*(alpha_r-1.0)

def calc_alpha_B(alpha_r,s):
    """
    Calculate constant alpha_gamma using eq. 10 of Soderberg et al. 2005
    """
    return ((2.0-s)/2.0)*alpha_r - 1.0

#--------------------------------------------------------------

# Calculate the characteristic frequencies as a function of time

def calc_nu_m(t,nu_m_0,t_0,alpha_gamma,alpha_B):
    """
    Calculate nu_m at t from above calcuated constants (see eq. A10 of Soderberg et al. 2005)
    """
    return nu_m_0*(t/t_0)**(2.0*alpha_gamma+alpha_B)

#--------------------------------------------------------------

# Caluclate the optical depth tau_nu at a given time 

def calc_tau_nu(t,t_0,C_tau,alpha_r,alpha_gamma,alpha_B,alpha_scrpitF,p,nu,F2):
    """
    Calculate tau_nu at t (see eq. A9 of Soderberg et al. 2005)
    """
    return (C_tau*((t/t_0)**((p-2.0)*alpha_gamma + (3.0+p/2.0)*alpha_B + alpha_r + alpha_scrpitF)) * (nu**(-(p+4.0)/2.0)) * F2).value

#--------------------------------------------------------------

# Finally calculate flux density

def calc_f_nu(t,t_0,C_f,alpha_r,alpha_B,tau_nu,zeta,p,nu,F2,F3):
    """
    Calculate f_nu at t (see eq. A8 of Soderberg et al. 2005)
    """
    eq1 = C_f*((t/t_0)**((4.0*alpha_r-alpha_B)/2.0))
    eq2 = ((1.0 - np.exp(-tau_nu**(zeta)))**(1.0/zeta)) * (nu**(5.0/2.0)) * F3/F2
    return ((eq1 * eq2).value * (u.erg / u.s /(u.cm)**2.0 /u.Hz)).to(u.mJy)

#--------------------------------------------------------------
#--------------------------------------------------------------

## Test Calculations ##


## Use the following lines of code to generate an SSA lightcurve with your own
## set of parameters.
#--------------------------------------------------------------

## Main physical parameters

d = (23.0 * u.Mpc).to(u.cm)       # cm ; distance to source
t_exp = 2453216.7                 # JD ; time of explosion
t_0 = 10                          # reference time 10 days since explosion
eta = 4                           # shell radius to thickness factor 
nu = 3.0e09 * u.Hz                # frequency of observation in Hz


B_0 = 1.06 * u.G                  # G
r_0 = 5.0e15 * u.cm               # cm
alpha_r = 0.9
p = 3.0
nu_m_0 = 0.02e9 * u.Hz            # Hz 
s = 2.0
zeta = 1.0

scriptF_0 = 1.0                   # as we have eps_e = eps_B
alpha_scrpitF = 0.0               # as we have eps_e = eps_B at all times

# Physical constants

m_e = const.m_e.cgs               # g
e = const.e.esu                   # esu
c = const.c.cgs                   # cm/s


#--------------------------------------------------------------

tstart = time.time()

t = np.arange(1,1001) 

alpha_gamma = calc_alpha_gamma(alpha_r)
alpha_B = calc_alpha_B(alpha_r,s)
gamma_m_0 = calc_gamma_m_0(B_0,nu_m_0)
C_tau = calc_C_tau(B_0,r_0,eta,gamma_m_0)
C_f = calc_C_f(B_0,r_0,d,p)
nu_m = calc_nu_m(t,nu_m_0,t_0,alpha_gamma,alpha_B)
x = (2.0/3.0) * (nu.value/nu_m.value)
F2 = calc_F_2(x,calc_F,p)
F3 = calc_F_3(x,calc_F,p)

## For large values of x, F2 and F3 do not vary as stated in Soderberg et al.
## 2005 but the integrator breaks. So replace the really small values of F2 and
## F3 with their last respective non-small value. 

#sel_large_x = np.where(x >= 22778.5451769)
#F2[sel_large_x] = F2[sel_large_x][0]
#F3[sel_large_x] = F3[sel_large_x][0]

## Finally calculate tau_nu and f_nu

tau_nu = calc_tau_nu(t,t_0,C_tau,alpha_r,alpha_gamma,alpha_B,alpha_scrpitF,p,nu,F2)
f_nu = calc_f_nu(t,t_0,C_f,alpha_r,alpha_B,tau_nu,zeta,p,nu,F2,F3)

## Write to table 

final_t = t + t_exp
final_f_nu = f_nu

final_tab = Table([final_t,f_nu,t,F2,F3],names=["t(JD)","f_nu(mJy)","epoch(days)","F2","F3"])
final_tab.write("3GHz_SN2004dk_lc_PySSA_test.txt",format="ascii",overwrite=True)

tend = time.time()
print("Time taken to print lc for one frequency : ", (tend - tstart))

##########

