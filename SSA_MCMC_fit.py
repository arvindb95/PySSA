## ---------------------- Import required packages ---------------------- ##

import numpy as np
import emcee
import astropy.units as u
from astropy.table import Table
from timeit import default_timer as timer
from PySSA import *
import scipy.optimize as op
from multiprocessing import Pool
import matplotlib as mpl

params = {"font.family": "serif", "text.usetex": True}

mpl.rcParams.update(params)

## ---------------------- Begin Code ---------------------- ##

## Main physical parameters ##

# Fixed params

# Held fixed during the fit. These reproduce Model 1 of Soderberg et al. 2005:
# a wind CSM (s=2) with constant eps_e/eps_B (scriptF_0=1, alpha_scriptF=0).
# to_interp=True is essential here - the exact quad path is ~1e4x slower and
# would make the MCMC intractable. Valid because p=3.2 is a grid node.
fixed_params = {
    "d": ((92.0 * u.Mpc).to(u.cm)).value,  # cm ; distance to source
    "t_0": 10,  # reference time 10 days since explosion
    "eta": 10,  # shell radius to thickness factor
    "p": 3.2,
    "nu_m_0": 1e9,  # Hz
    "scriptF_0": 1.0,  # as we have eps_e = eps_B
    "alpha_scriptF": 0.0,  # as we have eps_e = eps_B at all times
    "to_interp": True,  # Whether to use interpolation from saved grid to speed up calculations of F2 and F3 functions
    "s": 2.0,
    "xi": 0.5,
}

# Variable parameters

# Sampled parameters, with the values below used as the initial guess.
var_params = {
    "B_0": 3.0,  # G
    "r_0": 5.0e15,  # cm
    "alpha_r": 0.9,
}

var_params_names = list(var_params.keys())
print(var_params_names)
# r_0 is sampled as log10(r_0) because it spans many orders of magnitude.
# bounds and theta throughout are therefore in the TRANSFORMED space:
# bounds[1] constrains log10(r_0), not r_0.
param_scale_log = [False, True, False]
bounds = [(1.0e-50, np.inf), (1.0e-50, np.inf), (1.0e-50, 1.0)]

guess_params = np.array(list(var_params.values()))
guess_params[param_scale_log] = np.log10(guess_params[param_scale_log])

## ------------ Load data ------------##

comp_data = Table.read("./comprehensive_soderberg_data.csv", format="ascii.csv")

times = comp_data["Epoch"].data
freqs = comp_data["Freqs"].data
fluxes = comp_data["Fluxes"].data
fluxerrs = comp_data["Fluxerr"].data


## ------------ Define functions for MCMC ------------ ##


# Flat (improper) prior: 0 inside the bounds, -inf outside.
def lnprior(theta):
    allow_param_set = True

    for param_id in range(len(theta)):
        allow_param_set = allow_param_set and (
            bounds[param_id][0] <= theta[param_id] <= bounds[param_id][1]
        )
    if allow_param_set:
        return 0.0
    else:
        return -np.inf


# Gaussian likelihood. theta arrives in the transformed space, so undo the
# log10 scaling before handing parameters to the model.
def lnlike(theta, t, nu, F_obs, F_err):
    new_var_param_dict = {}

    for var_i in range(len(var_params_names)):
        if param_scale_log[var_i]:
            new_var_param_dict.update({var_params_names[var_i]: 10 ** (theta[var_i])})
        else:
            new_var_param_dict.update({var_params_names[var_i]: theta[var_i]})

    f_nu = SSA_flux_density(t, nu, **new_var_param_dict, **fixed_params)
    inv_sigma2 = 1.0 / F_err**2.0

    return -0.5 * (np.sum((F_obs - f_nu) ** 2 * inv_sigma2 - np.log(inv_sigma2)))


def lnprob(theta, t, nu, F_obs, F_err):
    lp = lnprior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, t, nu, F_obs, F_err)


# Small Gaussian ball around guess_parameters. Starting at the L-BFGS-B optimum
# rather than the raw initial guess roughly halves the autocorrelation time.
def get_starting_pos(guess_parameters, nwalkers, ndim):
    pos = [
        np.asarray(list(guess_parameters)) + 1e-2 * np.random.randn(ndim)
        for i in range(nwalkers)
    ]

    return pos


def run_mcmc(
    guess_parameters,
    pool,
    backend_file,
    niters,
    nwalkers,
    ndim,
    restart=False,
):
    t = times
    nu = freqs * 1e9  ### Make sure this is in Hz
    F = fluxes  ### Make sure this is in uJy
    F_err = fluxerrs  ### Make sure this is in uJy

    pos = get_starting_pos(guess_parameters, nwalkers, ndim=ndim)

    # HDF5 backend (needs h5py). restart=True resumes an existing chain in this
    # file instead of resetting it; run_mcmc(None, ...) then continues from the end.
    backend = emcee.backends.HDFBackend(backend_file)
    if restart == False:
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lnprob,
            args=(t, nu, F, F_err),
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
            args=(t, nu, F, F_err),
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


# Guard is required, not cosmetic: with the 'spawn' start method (macOS and
# Windows) each Pool worker re-imports this module. Without the guard every
# worker would rerun the minimisation and open its own Pool, spawning endlessly.
if __name__ == "__main__":
    ## ------------ Initial minimization ------------ ##

    # Cheap gradient-free-ish optimisation first, to give the walkers a good start.
    method = "L-BFGS-B"
    nll = lambda *args: -lnlike(*args)
    better_guess_params = op.minimize(
        nll,
        guess_params,
        bounds=bounds,
        args=(times, freqs * 1e9, fluxes, fluxerrs),
        method=method,
        options={"disp": True},
    )
    print("The minimized parameters using " + method)
    print(better_guess_params["x"])

    with Pool() as pool:
        sampler = run_mcmc(
            better_guess_params["x"],
            niters=10000,
            nwalkers=100,
            ndim=3,
            pool=pool,
            backend_file="SN2003L_model1_final.h5",
            restart=False,
        )
