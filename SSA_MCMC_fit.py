## ---------------------- Import required packages ---------------------- ##

import numpy as np
import emcee
import astropy.units as u
import astropy.constants as const
from astropy.table import Table
from timeit import default_timer as timer
from PySSA import *
import scipy.optimize as op
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib as mpl
from chainconsumer import ChainConsumer
import os

params = {
    'font.family': 'serif',
    'text.usetex': True}

mpl.rcParams.update(params)

## ---------------------- Begin Code ---------------------- ##

## Main physical parameters ##

d = ((23.0 * u.Mpc).to(u.cm)).value       # cm ; distance to source
t_exp = 2453216.7                         # JD ; time of explosion
t_0 = 10                                  # reference time 10 days since explosion
eta = 4                                   # shell radius to thickness factor
nu = 3.0e09                               # frequency of observation in Hz

B_0 = 0.5                    # G
r_0 = 5.0e15                 # cm
log_r_0 = np.log10(r_0)     
alpha_r = 0.9
p = 3.0
nu_m_0 = 0.02e9              # Hz
log_nu_m_0 = np.log10(nu_m_0)
s = 0.0
xi = 0.5

#guess_parameters = B_0, np.log10(r_0)#, alpha_r, p, np.log10(nu_m_0), s, xi

guess_parameters = B_0, s, xi

scriptF_0 = 1.0                   # as we have eps_e = eps_B
alpha_scrpitF = 0.0               # as we have eps_e = eps_B at all times

## Physical constants ##

m_e = (const.m_e.cgs).value               # g
e = (const.e.esu).value                   # esu
c = (const.c.cgs).value                   # cm/s


## ------------ Define functions for MCMC ------------ ##

def lnprior(theta):
    #B_0, log_r_0, alpha_r, p, log_nu_m_0, s, xi = theta

    #B_0, log_r_0 = theta
    B_0, s, xi = theta

    if (B_0 >= 1.0e-50) and (10**(log_r_0) >= 1.0e-50) and (1.0e-50 <= alpha_r <= 1.0) and (2.01 <= p <= 5.0) and (10**(log_nu_m_0) > 0.0) and (s >= 0.0) and (0.0 <= xi <= 1.0) :
        return 0.0
    else:
        return -np.inf

def lnlike(theta,t,t_0,nu,F_obs,F_err):
    #B_0, log_r_0, alpha_r, p, log_nu_m_0, s, xi = theta
    #B_0, log_r_0 = theta

    B_0, s, xi = theta
    
    alpha_gamma = calc_alpha_gamma(alpha_r)
    alpha_B = calc_alpha_B(alpha_r,s)
    gamma_m_0 = calc_gamma_m_0(B_0,10**(log_nu_m_0))
    C_tau = calc_C_tau(B_0,10**(log_r_0),eta,gamma_m_0,p,scriptF_0)
    C_f = calc_C_f(B_0,10**(log_r_0),d,p)

    nu_m = calc_nu_m(t,10**(log_nu_m_0),t_0,alpha_gamma,alpha_B)
    x = (2.0/3.0) * (nu/nu_m)
    F2 = calc_F_2(x,calc_F,p)
    F3 = calc_F_3(x,calc_F,p)
    
    tau_nu = calc_tau_nu(t,t_0,C_tau,alpha_r,alpha_gamma,alpha_B,alpha_scrpitF,p,nu,F2)
    f_nu = calc_f_nu(t,t_0,C_f,alpha_r,alpha_B,tau_nu,xi,p,nu,F2,F3)
    
    inv_sigma2 = 1.0/F_err**2.0 

    return -0.5*(np.sum((F_obs-f_nu)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprob(theta,t,t_0,nu,F_obs,F_err):
    lp = lnprior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta,t,t_0,nu,F_obs,F_err)

def get_starting_pos(guess_parameters, nwalkers, ndim=7):
    B_0 = guess_parameters[0]
    #log_r_0 = guess_parameters[1]
    #alpha_r = guess_parameters[2]
    #p = guess_parameters[3]
    #log_nu_m_0 = guess_parameters[4]
    #s = guess_parameters[5]
    #xi = guess_parameters[6]
    s = guess_parameters[1]
    xi = guess_parameters[2]


    #pos = [np.asarray([B_0, log_r_0, alpha_r, p, log_nu_m_0, s, xi]) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    #pos = [np.asarray([B_0, log_r_0]) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    pos = [np.asarray([B_0, s, xi]) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    return pos

def run_mcmc(data_table, guess_parameters, pool, backend_file, t_0 = 10, niters=500, nwalkers=200, ndim=7, restart=False):
    t = data_table['col1'].data
    nu = data_table['col2'].data * 10**9       ### Make sure this is in Hz
    F = data_table['col3'].data / 1000         ### Make sure this is in mJy
    F_err = data_table['col4'].data /1000      ### Make sure this is in mJy

    pos = get_starting_pos(guess_parameters,nwalkers, ndim=ndim)

    backend = emcee.backends.HDFBackend(backend_file)
    if (restart==False):
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, t_0, nu,F, F_err), backend=backend, pool=pool)
        
        print("## ------------ Starting MCMC run ------------ ##")

        start = timer()
        sampler.run_mcmc(pos, niters, progress=True)
        end = timer()

        print("Computation time: %f s"%(end-start))

        tau = backend.get_autocorr_time()
        print("The autocorrelation time for this run : ",tau)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,args=(t, t_0, nu,F, F_err), backend=backend,pool=pool)
        
        print("Initial size of chain in backend file: {0}".format(backend.iteration))
        
        print("## ------------ Starting MCMC run ------------ ##")

        start = timer()
        sampler.run_mcmc(None, niters, progress=True)
        end = timer()

        print("Computation time: %f s"%(end-start))
        print("Final size of chain in backend file: {0}".format(backend.iteration))

        tau = backend.get_autocorr_time()
        print("The autocorrelation time for this run : ",tau)

    return sampler


## ------------ Load data ------------##

#data_table = Table.read("SN2004dk_final_data_100days.txt",format="ascii")
data_table = Table.read("SN2004dk_final_data_later_times.txt", format="ascii")
t = data_table['col1'].data
t_0 = 10
nu = data_table['col2'].data * 10**9       ### Make sure this is in Hz
F = data_table['col3'].data / 1000         ### Make sure this is in mJy
F_err = data_table['col4'].data /1000      ### Make sure this is in mJy


## ------------ Initial minimization ------------ ##

#method = 'L-BFGS-B'

#nll = lambda *args: -lnlike(*args)
#bnds = [(1.0e-50,np.inf),(0.0,np.inf),(0.0,1.0)]
#better_guess_params = op.minimize(nll, guess_parameters, bounds=bnds, args=(t, t_0, nu,F, F_err), method=method)
#print("The minimized parameters using "+method)
#print(better_guess_params['x'])
#print(better_guess_params)

with Pool() as pool:
    sampler = run_mcmc(data_table, guess_parameters, niters=2000, nwalkers=10, ndim=3, pool=pool, backend_file="later_times_chain.h5",restart=True)
