import matplotlib.pyplot as plt
from astropy.table import Table
from mycolorpy import colorlist as mcp
import numpy as np
from PySSA import *

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

# SN 2003L radio photometry, transcribed from Table 1 of Soderberg et al. 2005.
# 108 VLA points at 4.9 / 8.5 / 15.0 / 22.5 GHz; epochs in days, fluxes in uJy.
# The single VLBA point in that table is deliberately excluded.
comp_data = Table.read("./comprehensive_soderberg_data.csv", format="ascii.csv")

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
        elinewidth=1,
        c=colors[i],
        fmt=".",
        label=str(uniq_freqs[i]) + r" GHz",
    )

## Borrow the physical parameters from Soderberg et al 2005 ()

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
    "to_interp": False,  # Whether to use interpolation from saved grid to speed up calculations of F2 and F3 functions
}
print(params)

# dict(params) makes an independent copy. Assigning `params_interp = params`
# would alias the same dict, so flipping to_interp below would also flip it in
# `params` and both curves plotted further down would be the interpolated one.
params_interp = dict(params)

params_interp["to_interp"] = True

print(params_interp)

# 3 free parameters (C_f, C_tau, alpha_r in the paper; B_0, r_0, alpha_r here),
# giving 105 dof, matching the value quoted in Soderberg et al. 2005.
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

## Main physical parameters ##

# as we have eps_e = eps_B at all times
# Model light curves from 10 to 1000 days after explosion.
t = np.logspace(1, 3, 100)

for i in range(len(uniq_freqs)):
    # Solid line: F2/F3 integrated from scratch with quad (slow, ~10 s per curve).
    # Dashed line: same model via the interpolated grid (~1e4x faster).
    # They should lie on top of each other - that agreement is the point of the plot.
    ssa_fnu = SSA_flux_density(t, uniq_freqs[i] * 1e9, **params)

    ax.plot(t, ssa_fnu, linewidth=1, color=colors[i])

    ssa_fnu_interp = SSA_flux_density(t, uniq_freqs[i] * 1e9, **params_interp)
    ax.plot(
        t,
        ssa_fnu_interp,
        linewidth=3,
        color=colors[i],
        linestyle="dashed",
        alpha=0.5,
    )

ax.legend(title="Frequency", ncol=2)
ax.set_xlabel("Time since burst (days)")
ax.set_ylabel(r"Flux density ($\mu$Jy)")

ax.set_xscale("log")
ax.set_yscale("log")


plt.savefig("Soderberg_2005_figure2.jpg", dpi=300)
plt.show()
