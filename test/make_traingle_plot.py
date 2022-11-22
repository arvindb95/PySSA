import numpy as np
from astropy.table import Table
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt

read_flat_samples = Table.read("flat_samples.txt",format="ascii")

col0 = read_flat_samples["col0"].data
col1 = read_flat_samples["col1"].data

flat_samples_array = np.vstack((col0[1000:],(col1[1000:]))).T

c = ChainConsumer()
c.add_chain(flat_samples_array, parameters=["$B_{{0}}$", "$r_{{0}}$"])

fig = c.plotter.plot(figsize="column")

plt.show()
