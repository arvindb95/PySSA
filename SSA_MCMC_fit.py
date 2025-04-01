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

params = {"font.family": "serif", "text.usetex": True}

mpl.rcParams.update(params)

## ---------------------- Begin Code ---------------------- ##

## Main physical parameters ##
params = {
    "d": ((92.0 * u.Mpc).to(u.cm)).value,  # cm ; distance to source
    "t_0": 10,  # reference time 10 days since explosion
    "eta": 10,  # shell radius to thickness factor
    "B_0": 4.5,  # G
    "r_0": 4.3e15,  # cm
    "alpha_r": 0.96,
    "p": 3.2,
    "nu_m_0": 1e9,  # Hz
    "s": 2.0,
    "xi": 0.5,
    "scriptF_0": 1.0,  # as we have eps_e = eps_B
    "alpha_scriptF": 0.0,  # as we have eps_e = eps_B at all times
}

# guess_parameters = B_0, np.log10(r_0)#, alpha_r, p, np.log10(nu_m_0), s, xi

guess_parameters = (
    params["B_0"],
    np.log10(params["r_0"]),
    params["alpha_r"],
    params["xi"],
)

d = params["d"]
t_0 = params["t_0"]
eta = params["eta"]
p = params["p"]
nu_m_0 = params["nu_m_0"]
s = params["s"]
scriptF_0 = params["scriptF_0"]
alpha_scriptF = params["alpha_scriptF"]

## Physical constants ##

m_e = (const.m_e.cgs).value  # g
e = (const.e.esu).value  # esu
c = (const.c.cgs).value  # cm/s

## ------------ Load data ------------##

comp_data = Table.read("comprehensive_soderberg_data.csv", format="ascii.csv")

times = comp_data["Epoch"].data
freqs = comp_data["Freqs"].data
fluxes = comp_data["Fluxes"].data
fluxerrs = comp_data["Fluxerr"].data


## ------------ Define functions for MCMC ------------ ##


def lnprior(theta):
    B_0, log_r_0, alpha_r, xi = theta

    if (
        (B_0 >= 1.0e-50)
        and (10 ** (log_r_0) >= 1.0e-50)
        and (1.0e-50 <= alpha_r <= 1.0)
        and (0.0 <= xi <= 1.0)
    ):
        return 0.0
    else:
        return -np.inf


def lnlike(theta, t, t_0, nu, F_obs, F_err):
    B_0, log_r_0, alpha_r, xi = theta

    f_nu = SSA_flux_density(
        t,
        nu,
        d,
        t_0,
        eta,
        B_0,
        10 ** (log_r_0),
        alpha_r,
        p,
        nu_m_0,
        s,
        xi,
        scriptF_0,
        alpha_scriptF,
    )

    inv_sigma2 = 1.0 / F_err**2.0

    new_tab = Table(
        [theta, f_nu],
        names=["theta", "f_nu"],
    )
    new_tab.write("store_flux_values.txt", format="ascii")

    return -0.5 * (np.sum((F_obs - f_nu) ** 2 * inv_sigma2 - np.log(inv_sigma2)))


def lnprob(theta, t, t_0, nu, F_obs, F_err):
    lp = lnprior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, t, t_0, nu, F_obs, F_err)


def get_starting_pos(guess_parameters, nwalkers, ndim=7):
    B_0, log_r_0, alpha_r, xi = guess_parameters

    # pos = [np.asarray([B_0, log_r_0, alpha_r, p, log_nu_m_0, s, xi]) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    # pos = [np.asarray([B_0, log_r_0]) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    pos = [
        np.asarray([B_0, log_r_0, alpha_r, xi]) + 1e-4 * np.random.randn(ndim)
        for i in range(nwalkers)
    ]
    return pos


def run_mcmc(
    guess_parameters,
    pool,
    backend_file,
    t_0=10,
    niters=500,
    nwalkers=200,
    ndim=7,
    restart=False,
):
    t = times
    nu = freqs * 1e9  ### Make sure this is in Hz
    F = fluxes  ### Make sure this is in mJy
    F_err = fluxerrs  ### Make sure this is in mJy

    pos = get_starting_pos(guess_parameters, nwalkers, ndim=ndim)

    backend = emcee.backends.HDFBackend(backend_file)
    if restart == False:
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lnprob,
            args=(t, t_0, nu, F, F_err),
            backend=backend,
            pool=pool,
        )

        print("## ------------ Starting MCMC run ------------ ##")

        start = timer()
        sampler.run_mcmc(pos, niters, progress=True)
        end = timer()

        print("Computation time: %f s" % (end - start))

        tau = backend.get_autocorr_time()
        print("The autocorrelation time for this run : ", tau)
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lnprob,
            args=(t, t_0, nu, F, F_err),
            backend=backend,
            pool=pool,
        )

        print("Initial size of chain in backend file: {0}".format(backend.iteration))

        print("## ------------ Starting MCMC run ------------ ##")

        start = timer()
        sampler.run_mcmc(None, niters, progress=True)
        end = timer()

        print("Computation time: %f s" % (end - start))
        print("Final size of chain in backend file: {0}".format(backend.iteration))

        tau = backend.get_autocorr_time()
        print("The autocorrelation time for this run : ", tau)

    return sampler


## ------------ Initial minimization ------------ ##

# method = "L-BFGS-B"
# nll = lambda *args: -lnlike(*args)
# bnds = [(1.0e-50, np.inf), (1.0e-50, np.inf), (1.0e-50, 1.0), (0.0, 1.0)]
# better_guess_params = op.minimize(
#    nll,
#    guess_parameters,
#    bounds=bnds,
#    args=(times, t_0, freqs * 1e9, fluxes, fluxerrs),
#    method=method,
#    options={"disp": True},
# )
# print("The minimized parameters using " + method)
# print(better_guess_params["x"])
# print(better_guess_params)

with Pool() as pool:
    sampler = run_mcmc(
        [4.473e00, 1.568e01, 9.543e-01, 3.808e-01],
        niters=10,
        nwalkers=10,
        ndim=4,
        pool=pool,
        backend_file="SN2003L.h5",
        restart=False,
    )
