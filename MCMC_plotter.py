import emcee
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from chainconsumer import (
    Chain,
    ChainConfig,
    ChainConsumer,
    PlotConfig,
    Truth,
    make_sample,
    truth,
)
import pandas as pd
from astropy.table import Table
from tqdm import tqdm


mpl.rcParams["font.size"] = 13
mpl.rcParams["legend.fontsize"] = 10

filename = "./SN2003L_model1_final.h5"
reader = emcee.backends.HDFBackend(filename)

tau = reader.get_autocorr_time()
print(tau)
burnin = int(2 * np.max(tau))

samples = reader.get_chain(discard=burnin, flat=True)
print(np.shape(samples))

samples[:, 1] = 10 ** (samples[:, 1])

# samples[:,1] = 10**samples[:,1]

ndim = 3

labels = [
    r"$B_{{0}}$",
    r"$r_{{0}}$",
    r"$\alpha_{{r}}$",
]

tab_labels = ["B_0", "r_0", "alpha_r"]

df = pd.DataFrame(samples, columns=labels)

c = ChainConsumer()
c.add_chain(Chain(samples=df, name="data table"))


best_fit_values = np.zeros(len(labels))
ll_values = np.zeros(len(labels))
ul_values = np.zeros(len(labels))

loc_best_fit = {}

for i in range(len(labels)):
    param_best_fit = c.analysis.get_summary()["data table"][labels[i]].center
    param_ll = c.analysis.get_summary()["data table"][labels[i]].lower
    param_ul = c.analysis.get_summary()["data table"][labels[i]].upper

    best_fit_values[i] = param_best_fit
    ll_values[i] = param_ll
    ul_values[i] = param_ul
    loc_best_fit.update({labels[i]: param_best_fit})


mcmc_best_fit_tab = Table(
    [
        tab_labels,
        best_fit_values,
        ll_values,
        ul_values,
    ],
    names=["param", "best_fit", "ll", "ul"],
)


mcmc_best_fit_tab.write(
    "mcmc_best_fit_params_model1_final.txt", format="ascii", overwrite=True
)

# Now plotting

c.set_plot_config(
    PlotConfig(
        usetex=True,
        label_font_size=20,
        tick_font_size=15,
        contour_label_font_size=20,
        summary_font_size=18,
        sigma2d=True,
    )
)



c.add_truth(Truth(location=loc_best_fit))

c.plotter.plot()
plt.savefig("best_fit_corner_plot_model1_final.jpg", dpi=300)

c.plotter.plot_walks(convolve=100, plot_weights=False)
plt.savefig("walks_model1_final.jpg", dpi=300)
