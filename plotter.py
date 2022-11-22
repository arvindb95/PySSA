import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import emcee
from chainconsumer import ChainConsumer
from PySSA import *

params = {
    'font.family': 'serif',
    'text.usetex': True}

mpl.rcParams.update(params)

filename="first_100_days_chain_new_lnlike.h5"

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

ndim = 2

fig, axes = plt.subplots(ndim, figsize=(10, 14), sharex=True)
labels=[r"$B_{{0}}$",r"$r_{{0}}$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, i], "C0", alpha=0.7)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.axvline(burnin,color="r")
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

plt.savefig("parameter_variations.pdf")

c = ChainConsumer()
c.add_chain(samples, parameters=[r"$B_{{0}}$", r"$log_{{10}}(r_{{0}}$)"],usetex=True)

fig2 = c.plotter.plot(figsize=(8,7))

plt.savefig("corner_test.pdf")

