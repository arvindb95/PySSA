import matplotlib.pyplot as plt
from astropy.table import Table
from mycolorpy import colorlist as mcp
import numpy as np
from PySSA import *
from tqdm import tqdm
import emcee

import time

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

comp_data = Table.read("comprehensive_soderberg_data.csv", format="ascii.csv")

times = comp_data["Epoch"].data
freqs = comp_data["Freqs"].data
fluxes = comp_data["Fluxes"].data
fluxerrs = comp_data["Fluxerr"].data

uniq_freqs = np.sort(np.unique(freqs))
colors = mcp.gen_color_normalized("viridis", data_arr=uniq_freqs)

fig = plt.figure()

ax = fig.add_subplot(111)

for i in range(len(uniq_freqs)):
    sel_freq = np.where(freqs == uniq_freqs[i])
    ax.errorbar(
        times[sel_freq],
        fluxes[sel_freq],
        yerr=fluxerrs[sel_freq],
        capsize=2,
        elinewidth=0.5,
        c=colors[i],
        fmt=".",
        label=str(uniq_freqs[i]) + r" GHz",
    )

## Borrow the physical parameters from Soderberg et al 2005 ()

t_range = np.logspace(1, 3, 100)

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
    "to_interp": True,  # Whether to use interpolation from saved grid to speed up calculations of F2 and F3 functions
}

best_fit_tab = Table.read("./mcmc_best_fit_params_model1_final.txt", format="ascii")

best_fit_params = best_fit_tab["param"].data
best_fit_vals = best_fit_tab["best_fit"].data
ll_values = best_fit_tab["ll"].data
ul_values = best_fit_tab["ul"].data


for i in range(len(best_fit_params)):
    if "log_" in best_fit_params[i]:
        params.update({(best_fit_params[i].split("log_")[1]): 10 ** (best_fit_vals[i])})
    else:
        params.update({best_fit_params[i]: best_fit_vals[i]})

## Calculate chi_sq

n_of_obs = len(times)
n_of_fit_params = 3

dof = n_of_obs - n_of_fit_params


def calc_chi_sq(flux, flux_errs, model):
    return np.sum(((flux - model) ** 2) / (flux_errs**2))


model = SSA_flux_density(t=times, nu=freqs * 1e9, **params)

chi_sq = calc_chi_sq(fluxes, fluxerrs, model)

print("###############################")
print("Degrees of freedom = ", dof)
print("Chi_sq = ", chi_sq)
print("Reduced chi_sq = ", chi_sq / dof)
print("###############################")

t0 = time.time()
# Calculate errors on F_nu
filename = "./SN2003L_model1_final.h5"
reader = emcee.backends.HDFBackend(filename)

tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
samples = reader.get_chain(discard=burnin, flat=True)

print(np.shape(samples))

samples[:, 1] = 10 ** (samples[:, 1])

sel_best_fit_values = np.ones(len(samples)).astype("bool")

labels = [r"$B_{{0}}$", r"$r_{{0}}$", r"$\alpha_{{r}}$"]

tab_labels = ["B_0", "r_0", "alpha_r"]


for i in range(len(labels)):
    sel_best_fit_values &= (ll_values[i] < samples[:, i]) & (
        samples[:, i] < ul_values[i]
    )

print(np.unique(sel_best_fit_values))

final_param_set = samples


print("Calculating errors on F_nu")

f_nu_ll_list = []
f_nu_ul_list = []
freq_list = []
t_list = []

t_r = np.logspace(1, 3, 25)

ssa_fnu_all = np.zeros((len(final_param_set), len(t_r), len(uniq_freqs)))

for i, nu_i in enumerate(uniq_freqs):
    for p in tqdm(range(len(final_param_set))):
        params.update(
            {
                "B_0": final_param_set[p][0],
                "r_0": final_param_set[p][1],
                "alpha_r": final_param_set[p][2],
            }
        )
        ssa_fnu_all[p, :, i] = SSA_flux_density(t=t_r, nu=nu_i * 1e9, **params)

f_nu_ll_list = np.min(ssa_fnu_all, axis=0)
f_nu_ul_list = np.max(ssa_fnu_all, axis=0)

center_values = np.percentile(
    ssa_fnu_all, [3e-5, 0.149999, 2.2999, 16, 50, 84, 97.7, 99.85, 99.9999699], axis=0
)
# q_values = np.diff(center_values, axis=0)

print(center_values[:, 1, 1])

for i in tqdm(range(len(uniq_freqs))):
    ssa_fnu = SSA_flux_density(t=t_range, nu=uniq_freqs[i] * 1e9, **params)
    ax.plot(t_range, ssa_fnu, linewidth=0.5, color=colors[i], zorder=10)
    # plotting 5 sigma error on flux
    ax.fill_between(
        t_r,
        y1=center_values[0, :, i],
        y2=center_values[8, :, i],
        color=colors[i],
        zorder=-1,
        alpha=0.3,
    )
    """
    ax.fill_between(
        t_r,
        y1=center_values[1, :, i],
        y2=center_values[7, :, i],
        color="k",
        zorder=-1,
        label="3 sigma",
    )

    ax.fill_between(
        t_r,
        y1=center_values[2, :, i],
        y2=center_values[6, :, i],
        color="grey",
        zorder=-1,
        label="2 sigma",
    )
    ax.fill_between(
        t_r,
        y1=center_values[3, :, i],
        y2=center_values[5, :, i],
        color="lightgrey",
        zorder=-1,
        label="1 sigma",
    )
    """

ax.legend(title="Frequency", ncol=2)
ax.set_xlabel("Time since burst (days)")
ax.set_ylabel(r"Flux density ($\mu$Jy)")

ax.set_xscale("log")
ax.set_yscale("log")
plt.savefig("Soderberg_2005_figure2_SSA_fit_model1_with_5sigma_err.jpg", dpi=300)
plt.show()
