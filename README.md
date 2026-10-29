# Forbidden Formation Histories

## Overview

The BBH merger rate is the progenitor formation rate convolved with the delay time distribution. This code solves that convolution backwards: given a merger rate and an assumed delay time distribution, it computes the progenitor formation rate directly in the time domain, without assuming the formation rate's functional form. A delay time distribution that requires a negative formation rate is incompatible with the merger rate, so the method can test any delay time distribution, including redshift-dependent predictions from population synthesis, against a merger rate posterior.

This is companion code to the paper "Forbidden Formation Histories: The Binary Black Hole Merger Rate Disfavours Long Delay Times" ([arXiv:2605.12858](https://arxiv.org/abs/2605.12858)), and the notebook makes every result and figure of the paper.

## Using the method

```python
import numpy as np
from astropy import units as u
from deconvolution import cosmo, Grid, solve

# zs: redshifts of the merger rate posterior, R: merger rate draws, shape (len(zs), n_draws) [Gpc^-3 yr^-1]
grid = Grid(cosmo.age(zs).to_value(u.yr))
Rm = grid.merger_rate(R, boundary=0)          # merger rate beyond z = 1.5 [Gpc^-3 yr^-1]

# your delay time distribution, tabulated in Gyr
t = np.logspace(-2, np.log10(13.7), 500)
p = t**-1.0
K = grid.forward_matrix(grid.binned_kernel(t, p))

R_form = solve(K, Rm)                         # formation rate of every draw on the grid
P = grid.compatibility(K, Rm)                 # fraction of draws with a non-negative formation rate
```

For a delay time distribution that depends on formation time, pass one kernel per metallicity class and their weights at each formation step to `grid.forward_matrix` (see `compas_delay_times.py`).

## Files

* `deconvolution_mergerrate.ipynb`: the analysis, top to bottom (`deconvolution_mergerrate.html` is a rendered copy with outputs).
* `deconvolution.py`: the solver grid, binned kernels, triangular solve, compatibility and helper functions.
* `compas_delay_times.py`: reads the COMPAS release and builds p(tau | t_form) from the van Son et al. (2023) metallicity distribution of star formation and the COMPAS yields.
* `make_processed_delay_times.py`: writes the processed delay time distributions p(tau | z_form) of all 35 COMPAS models (fiducial metallicity history) to `processed_delay_times.h5`, which is included.
* `data/`: the input data, with sources and formats in `data/README.md`.
* `figures/`: the paper figures made by the notebook: Fig. 1 `quota.png`, Fig. 2 `formationrates_withfails.png`, Fig. 3 `alpha_v_kappa.png`, Fig. 4 `passrate_vs_alpha.png`, Fig. 5 `popsynth.png`, Fig. 6 `merger_rates.png`.
* `CITATION.cff`: citation metadata for this code.

## Data

The input data are included in `data/`, redistributed under CC BY 4.0 from their original releases (see `data/README.md` for details and changes). If you use them, please cite:

* GWTC-4.0 BBH merger rate (`data/Rates_redshift_O4_bspline.h5`): LIGO Scientific Collaboration, Virgo Collaboration & KAGRA Collaboration (2025), "GWTC-4.0: Population Properties of Merging Compact Binaries", https://doi.org/10.5281/zenodo.16911563.
* COMPAS delay time distributions and BBH yields (`data/10019887/Output/`): Fishbach & van Son (2023), "LIGO-Virgo-KAGRA's Oldest Black Holes: Probing star formation at cosmic noon with GWTC-3", https://doi.org/10.5281/zenodo.10019887, from the COMPAS simulations of van Son et al. (2022).

The metallicity distribution of star formation (van Son et al. 2023) and the star formation rate shape (Katsianis et al. 2021) are analytic and are coded in `compas_delay_times.py` and the notebook.

## Running

Run `deconvolution_mergerrate.ipynb` from this folder (about two minutes on a laptop). To write the processed delay time distributions:

```
python make_processed_delay_times.py
```

`processed_delay_times.h5` has the delay time grid `tau` [Gyr], the formation redshift grid `z_form`, and one group per COMPAS model with `p_tau` (z_form, tau) [Gyr^-1], the metallicity class weights `weights` and the class distributions `p_tau_class`. Units and the metallicity parameters are stored as attributes.

## Citation

If you use this code, please cite the paper, [arXiv:2605.12858](https://arxiv.org/abs/2605.12858), and this software release, [doi:10.5281/zenodo.23773173](https://doi.org/10.5281/zenodo.23773173) (see `CITATION.cff`).

## Dependencies

Python 3 with the packages in `requirements.txt` (tested with Python 3.11, numpy 1.24, scipy 1.11, matplotlib 3.7, seaborn 0.12, astropy 6.0, h5py 3.9):

```
pip install -r requirements.txt
```
