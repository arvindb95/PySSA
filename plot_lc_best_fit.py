import matplotlib.pyplot as plt
from astropy.table import Table
from mycolorpy import colorlist as mcp
import numpy as np
from PySSA import *
from tqdm import tqdm
import emcee

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
}

best_fit_tab = Table.read("mcmc_best_fit_params.txt", format="ascii")

best_fit_params = best_fit_tab["param"].data
best_fit_vals = best_fit_tab["best_fit"].data
ll_values = best_fit_tab["ll"].data
ul_values = best_fit_tab["ul"].data


for i in range(len(best_fit_params)):
    if "log_" in best_fit_params[i]:
        params.update({(best_fit_params[i].split("log_")[1]): 10 ** (best_fit_vals[i])})
    else:
        params.update({best_fit_params[i]: best_fit_vals[i]})

# Calculate errors on F_nu
filename = "SN2003L.h5"
reader = emcee.backends.HDFBackend(filename)

tau = reader.get_autocorr_time()
burnin = int(1 * np.max(tau))
samples = reader.get_chain(discard=burnin, flat=True)

sel_best_fit_values = np.ones(len(samples)).astype("bool")

labels = [r"$B_{{0}}$", r"$\rm{{log}}_{{10}}r_{{0}}$", r"$\alpha_{{r}}$", r"$\xi$"]

tab_labels = ["B_0", "log_r_0", "alpha_r", "xi"]

for i in range(len(labels)):
    sel_best_fit_values &= (ll_values[i] < samples[:, i]) & (
        samples[:, i] < ul_values[i]
    )

final_param_set = samples[sel_best_fit_values, :]

print("Calculating errors on F_nu")

f_nu_ll_list = []
f_nu_ul_list = []
freq_list = []
t_list = []

#for t_r in np.logspace(1, 3, 25):
#    print("===============for t = ", t_r)
#    for nu_i in uniq_freqs:
#        f_nu = []
#        for p in tqdm(range(len(final_param_set))):
#            params.update(
#                {
#                    "B_0": final_param_set[p][0],
#                    "r_0": 10 ** (final_param_set[p][1]),
#                    "alpha_r": final_param_set[p][2],
#                    "xi": final_param_set[p][3],
#                }
#            )
#            f_nu.append(SSA_flux_density(t=t_r, nu=nu_i * 1e9, **params))
#        f_nu_ll_list.append(min(f_nu))
#        f_nu_ul_list.append(max(f_nu))
#        t_list.append(t_r)
#        freq_list.append(nu_i)
#
#
#error_table = Table(
#    [t_list, freq_list, f_nu_ll_list, f_nu_ul_list],
#    names=["times", "freqs", "f_nu_ll", "f_nu_ul"],
#)


error_table = Table.read('error_table.txt', format="ascii")

error_times = error_table['times'].data
error_freqs = error_table['freqs'].data
f_nu_ll = error_table['f_nu_ll'].data
f_nu_ul = error_table['f_nu_ul'].data

for i in tqdm(range(len(uniq_freqs))):
    
    sel_error_data = np.where(error_freqs==uniq_freqs[i])
    
    ssa_fnu = SSA_flux_density(t=t_range, nu=uniq_freqs[i] * 1e9, **params)
    ax.plot(t_range, ssa_fnu, linewidth=0.5, color=colors[i])
    ax.fill_between(error_times[sel_error_data], f_nu_ll[sel_error_data], f_nu_ul[sel_error_data], color=colors[i], alpha=0.3, edgecolor='none')

    


ax.legend(title="Frequency", ncol=2)
ax.set_xlabel("Time since burst (days)")
ax.set_ylabel(r"Flux density ($\mu$Jy)")

ax.set_xscale("log")
ax.set_yscale("log")
plt.savefig("Soderberg_2005_figure2_SSA_fit.pdf")
