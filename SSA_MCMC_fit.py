import numpy as np
import emcee
import corner
import astropy.units as u
import astropy.constants as const
from astropy.table import Table
from timeit import default_timer as timer
from PySSA import *
import matplotlib.pyplot as plt

import matplotlib as mpl

params = {
    'font.family': 'serif',
    'text.usetex': True}

mpl.rcParams.update(params)

## Main physical parameters

d = ((23.0 * u.Mpc).to(u.cm)).value       # cm ; distance to source
t_exp = 2453216.7                         # JD ; time of explosion
t_0 = 10                                  # reference time 10 days since explosion
eta = 4                                   # shell radius to thickness factor
nu = 3.0e09                               # frequency of observation in Hz

B_0 = 1.06                   # G
r_0 = 5.0e15                 # cm
alpha_r = 0.9
p = 3.0
nu_m_0 = 0.02e9              # Hz
s = 2.0
zeta = 1.0

guess_parameters = B_0, np.log10(r_0), alpha_r, p, np.log10(nu_m_0), s, zeta

scriptF_0 = 1.0                   # as we have eps_e = eps_B
alpha_scrpitF = 0.0               # as we have eps_e = eps_B at all times

# Physical constants

m_e = (const.m_e.cgs).value               # g
e = (const.e.esu).value                   # esu
c = (const.c.cgs).value                   # cm/s

#--------------------------------------------------------------

#tstart = time.time()

#alpha_gamma = calc_alpha_gamma(alpha_r)
#alpha_B = calc_alpha_B(alpha_r,s)
#gamma_m_0 = calc_gamma_m_0(B_0,nu_m_0)
#C_tau = calc_C_tau(B_0,r_0,eta,gamma_m_0,p,scriptF_0)
#C_f = calc_C_f(B_0,r_0,d,p)

### Finally calculate tau_nu and f_nu
#
#tau_nu = calc_tau_nu(t,t_0,C_tau,alpha_r,alpha_gamma,alpha_B,alpha_scrpitF,p,nu,F2)
#f_nu = calc_f_nu(t,t_0,C_f,alpha_r,alpha_B,tau_nu,zeta,p,nu,F2,F3)
#
### Write to table 
#
#final_t = t + t_exp
#final_f_nu = f_nu
#
#final_tab = Table([final_t,f_nu,t,F2,F3],names=["t(JD)","f_nu(mJy)","epoch(days)","F2","F3"])
#final_tab.write("3GHz_SN2004dk_lc_PySSA_test.txt",format="ascii",overwrite=True)
#
#tend = time.time()
#print("Time taken to print lc for one frequency : ", (tend - tstart))
#
###########

def lnprior(theta):
    B_0, log_r_0, alpha_r, p, log_nu_m_0, s, zeta = theta

    if (B_0 >= 1.0e-50) and (10**(log_r_0) >= 1.0e-50) and (1.0e-50 <= alpha_r <= 1.0) and (2.01 <= p <= 5.0) and (10**(log_nu_m_0) > 0.0) and (s >= 0.0) and (0.0 <= zeta <= 1.0) :
        return 0.0
    else:
        return -np.inf

def lnlike(theta,t,t_0,nu,F_obs,F_err):
    B_0, log_r_0, alpha_r, p, log_nu_m_0, s, zeta = theta

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
    f_nu = calc_f_nu(t,t_0,C_f,alpha_r,alpha_B,tau_nu,zeta,p,nu,F2,F3)
    
    inv_sigma2 = 1.0/F_err**2.0 

    return -0.5*(np.sum((F_obs-f_nu)**2*inv_sigma2 - 2*np.log(inv_sigma2)))

def lnprob(theta,t,t_0,nu,F_obs,F_err):
    lp = lnprior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta,t,t_0,nu,F_obs,F_err)

def get_starting_pos(guess_parameters, nwalkers, ndim=7):
    B_0 = guess_parameters[0]
    log_r_0 = guess_parameters[1]
    alpha_r = guess_parameters[2]
    p = guess_parameters[3]
    log_nu_m_0 = guess_parameters[4]
    s = guess_parameters[5]
    zeta = guess_parameters[6]

    pos = [np.asarray([B_0, log_r_0, alpha_r, p, log_nu_m_0, s, zeta]) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

    return pos

def run_mcmc(data_table, guess_parameters, t_0 = 10 ,niters=500, nthreads=1, nwalkers=200, ndim=7):
    t = data_table['col1'].data
    nu = data_table['col2'].data * 10**9       ### Make sure this is in Hz
    F = data_table['col3'].data / 1000         ### Make sure this is in mJy
    F_err = data_table['col4'].data /1000      ### Make sure this is in mJy

    pos = get_starting_pos(guess_parameters,nwalkers, ndim=ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, t_0, nu,F, F_err),threads=nthreads)

    start = timer()
    sampler.run_mcmc(pos, niters, progress=True)
    end = timer()

    print("Computation time: %f s"%(end-start))

    return sampler


data_table = Table.read("SN2004dk_final_data_100days_test.txt",format="ascii")

sampler = run_mcmc(data_table, guess_parameters,niters=100,nwalkers=20)

labels = [r"$B_{{0}}$",r"log$_{{10}}(r_{{0}})$",r"$\alpha_{{\rm{r}}}$", r"$p$", r"log$_{{10}}(\nu_{{\rm{m},0}})$", r"$s$", r"$\zeta$"]

#flat_samples = sampler.get_chain(flat=True)

ndim = 7

fig, axes = plt.subplots(7, figsize=(10, 14), sharex=True)
samples = sampler.get_chain()
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")


plt.savefig("parameter_valriations.pdf")

