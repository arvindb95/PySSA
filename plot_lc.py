import matplotlib.pyplot as plt
from astropy.table import Table
from mycolorpy import colorlist as mcp
import numpy as np
from PySSA import *
import pandas as pd

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
        elinewidth=1,
        c=colors[i],
        fmt=".",
        label=str(uniq_freqs[i]) + r" GHz",
    )

## Borrow the physical parameters from Soderberg et al 2005 ()

## Main physical parameters ##

d = ((92.0 * u.Mpc).to(u.cm)).value  # cm ; distance to source
t_0 = 10  # reference time 10 days since explosion
eta = 10  # shell radius to thickness factor

B_0 = 4.5  # G
r_0 = 4.3e15  # cm
alpha_r = 0.96
p = 3.2
nu_m_0 = 1e9  # Hz
log_nu_m_0 = np.log10(nu_m_0)
s = 2.0
xi = 0.5

scriptF_0 = 1.0  # as we have eps_e = eps_B
alpha_scriptF = 0.0  # as we have eps_e = eps_B at all times
t = np.logspace(1, 3, 100)


for i in range(len(uniq_freqs)):
    ssa_fnu = SSA_flux_density(
        t,
        t_0,
        uniq_freqs[i] * 1e9,
        d,
        eta,
        B_0,
        r_0,
        alpha_r,
        p,
        nu_m_0,
        s,
        xi,
        scriptF_0,
        alpha_scriptF,
    )
    ax.plot(t, ssa_fnu, linewidth=1, color=colors[i])

ax.legend(title="Frequency", ncol=2)
ax.set_xlabel("Time since burst (days)")
ax.set_ylabel(r"Flux density ($\mu$Jy)")

ax.set_xscale("log")
ax.set_yscale("log")
plt.savefig("Soderberg_2005_figure2.jpg", dpi=300)
