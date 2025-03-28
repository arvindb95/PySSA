# PySSA  
Python implementation of the Sychrotron Self Absorbed (SSA) model from
[Soderberg et al. 2005](https://ui.adsabs.harvard.edu/abs/2005ApJ...621..908S/abstract)

## Usage 
- Use `PySSA.py` to generate an SSA lightcurve for your choice of parameters. 
- The `plot_lc.py` code allows you to make the lightcurve  plot. Here the example shown is to reproduce figure 2 of the above paper. 
- The `comprehensive_soderberg_data.csv` file contains the relevant data from the above paper to plot the light curve for supernova 2003L. 

![Here is the output of `plot_lc.py`. Compare this with Figure 2 of [Soderberg et al. 2005](https://ui.adsabs.harvard.edu/abs/2005ApJ...621..908S/abstract)](Soderberg_2005_figure2.jpg "SN 2003L lightcurve")

## To do

- Make calculations of integrals faster in `PySSA.py`
- Add MCMC fitting script
- Add plotter scripts for visualysing MCMC fits using [ChainConsumer](https://samreay.github.io/ChainConsumer/)
- Add list of dependancies 


