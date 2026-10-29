# Data

The data here are redistributed from their original releases under the Creative Commons Attribution 4.0 International license (CC BY 4.0). If you use them, please cite the original releases below.

## GWTC-4.0 BBH merger rate

`Rates_redshift_O4_bspline.h5`: the 5000 posterior draws of the BBH merger rate as a function of redshift from the GWTC-4.0 B-Spline population model (the `rate_vs_redshift` draws in `BBHMassSpinRedshift_BSplineIID.h5`, in `analyses_BBH.tar`), stored as `zs` (redshift, 634 points) and `R_zs` (merger rate, shape (634, 5000), in Gpc^-3 yr^-1).

Changes: we keep the first 634 of the 1000 redshift points of the original grid (z <= 1.5, the range used in the paper) and store the draws as (redshift, draw); the values are unchanged.

Source: LIGO Scientific Collaboration, Virgo Collaboration & KAGRA Collaboration (2025), "GWTC-4.0: Population Properties of Merging Compact Binaries", Zenodo, https://doi.org/10.5281/zenodo.16911563 (CC BY 4.0).

## COMPAS delay time distributions and BBH yields

`10019887/Output/`: the contents of `Output.zip`, unchanged.

* `histogram_data_<model>_LOG.csv`: delay time distribution of one COMPAS model, in bins of 0.2 dex. Columns come in pairs, `<group>_xvalues` (log10 of the delay time in Gyr, bin centre) and `<group>_yvalues` (number of merging BBHs per unit log10 delay time per 10^6 Msun of star formation; not normalised, the code normalises each distribution), for the groups `All`, `CE` (common envelope), `stable` (stable mass transfer), `lowZ` (Z <= Zsun/10), `midZ` (Zsun/10 < Z <= Zsun/2) and `highZ` (Z > Zsun/2), with Zsun = 0.014. We use the three metallicity classes.
* `Yield_data.h5`: one record per model over 14 metallicity bins, including the fields `Metallicity_bin` (log10 Z), `merging_BBH`, `Average_SF_mass_needed` and `mixture_weight_sum`. The BBH yield per unit star-forming mass [Msun^-1] is `merging_BBH / Average_SF_mass_needed / mixture_weight_sum`.

Source: Fishbach, M. & van Son, L. (2023), "Data Release: LIGO-Virgo-KAGRA's Oldest Black Holes: Probing star formation at cosmic noon with GWTC-3", Zenodo, https://doi.org/10.5281/zenodo.10019887 (CC BY 4.0). The simulations are those of van Son et al. (2022).

Changes: none.
