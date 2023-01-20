import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import emcee
from chainconsumer import ChainConsumer
from PySSA import *
import corner

#params = {
#    'font.family': 'serif',
#    'text.usetex': True}

#mpl.rcParams.update(params)

## ---------------------- Begin Code ---------------------- ##

## Main physical parameters ##

d = ((23.0 * u.Mpc).to(u.cm)).value       # cm ; distance to source
t_exp = 2453216.7                         # JD ; time of explosion
t_0 = 10                                  # reference time 10 days since explosion
eta = 4                                   # shell radius to thickness factor
nu = 3.0e09                               # frequency of observation in Hz

B_0 = 1.0                    # G
r_0 = 5.0e15                 # cm
log_r_0 = np.log10(r_0)
alpha_r = 0.9
p = 3.0
nu_m_0 = 0.02e9              # Hz
log_nu_m_0 = np.log10(nu_m_0)
s = 2.0
xi = 1.0

scriptF_0 = 1.0                   # as we have eps_e = eps_B
alpha_scriptF = 0.0               # as we have eps_e = eps_B at all times


guess_parameters = B_0, np.log10(r_0)#, alpha_r, p, np.log10(nu_m_0), s, xi

#guess_parameters = B_0, s, xi

scriptF_0 = 1.0                   # as we have eps_e = eps_B
alpha_scrpitF = 0.0               # as we have eps_e = eps_B at all times

## Physical constants ##

m_e = (const.m_e.cgs).value               # g
e = (const.e.esu).value                   # esu
c = (const.c.cgs).value                   # cm/s


filename="first_100_days_chain.h5"

reader = emcee.backends.HDFBackend(filename)

tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
#log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
#print("flat log prob shape: {0}".format(log_prob_samples.shape))
#print("flat log prior shape: {0}".format(log_prior_samples.shape))

#ndim = 3
ndim = 2

B_best_fit = np.percentile(samples[:,0], [16,50,84])
s_best_fit = 10**(np.percentile(samples[:,1], [16,50,84]))
#xi_best_fit = np.percentile(samples[:,2], [16,50,84])
print(B_best_fit)
print(s_best_fit)
#print(xi_best_fit)

print(len(samples[:,0]))

#fig, axes = plt.subplots(ndim, figsize=(10, 14), sharex=True)
#labels=[r"$B_{{0}}$",r"$s$",r"$\xi$"]
labels=[r"$B_{{0}}$",r"$log_{{10}}r_{{0}}$"]
#for i in range(ndim):
#    ax = axes[i]
#    ax.plot(samples[:, i], "C0", alpha=0.7)
#    ax.set_xlim(0, len(samples))
#    ax.set_ylabel(labels[i])
#    ax.axvline(burnin,color="r")
#    ax.yaxis.set_label_coords(-0.1, 0.5)

#axes[-1].set_xlabel("step number")

#plt.savefig("parameter_variations_later_times.pdf")

c = ChainConsumer()
c.add_chain(samples, parameters=labels)
c.configure(colors=["#ff7f0e"], bar_shade=True)
#c.configure(statistics="cumulative")

B_best_fit = c.analysis.get_summary()[labels[0]][1]
r_best_fit = c.analysis.get_summary()[labels[1]][1]
#s_best_fit = c.analysis.get_summary()[labels[1]][1]
#xi_best_fit = c.analysis.get_summary()[labels[2]][1]

print("B best fit : ",B_best_fit)
print("r best fit : ",10**r_best_fit)

#fig2 = c.plotter.plot(figsize=(8,7), truth=[B_best_fit,s_best_fit,xi_best_fit])

fig2 = c.plotter.plot(figsize=(8,7), truth=[B_best_fit,r_best_fit])

plt.savefig("corner_test_later_timesi_maxlik.pdf")

data_table = Table.read("SN2004dk_final_data_100days.txt",format="ascii")
t_obs = data_table['col1'].data
t_0 = 10
nu = data_table['col2'].data * 10**9       ### Make sure this is in Hz
F = data_table['col3'].data / 1000         ### Make sure this is in mJy
F_err = data_table['col4'].data /1000      ### Make sure this is in mJy

#t = np.logspace(5,6,100)

t = np.logspace(0.5, 3, 100)

ssa_fnu = SSA_flux_density(t,t_0,3e9,d,eta,B_best_fit,10**(r_best_fit),alpha_r,p,nu_m_0,s,xi,scriptF_0,alpha_scriptF)

fig3 = plt.figure()

plt.plot(t,ssa_fnu,c="C1")
plt.yscale("log")
plt.xscale("log")
plt.savefig("first_100days_lc.pdf")

def calc_chi_sq(obs_flux,obs_flux_err,obs_time,obs_freq,t_0,d,eta,B_0,r_0,alpha_r,p,nu_m_0,s,xi,scriptF_0,alpha_scriptF):
    model_flux = SSA_flux_density(obs_time,t_0,obs_freq,d,eta,B_0,r_0,alpha_r,p,nu_m_0,s,xi,scriptF_0,alpha_scriptF)
    chi_sq = np.sum((obs_flux-model_flux)**2/(obs_flux_err**2))
    return chi_sq


print("Chi_sq : ",calc_chi_sq(F,F_err,t_obs,nu,t_0,d,eta,B_best_fit,10**(r_best_fit),alpha_r,p,nu_m_0,s,xi,scriptF_0,alpha_scriptF))
