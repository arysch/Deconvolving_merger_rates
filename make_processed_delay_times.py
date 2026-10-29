"""
Write the redshift-dependent COMPAS delay time distributions p(tau | z_form) used in the paper to one HDF5 file.

For each of the 35 COMPAS models of Fishbach & van Son (2023) with per-metallicity yields, the file holds
p(tau | z_form) = sum_k w_k(z_form) p_k(tau) (paper Eq. 6) for the fiducial metallicity history of
van Son et al. (2023), normalised to unit probability between tau_min = 10 Myr and the age of the Universe.

Usage:
    python make_processed_delay_times.py [output file]     (default: processed_delay_times.h5)
"""
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import trapezoid

from deconvolution import TAU_MIN, TAU_MAX
from compas_delay_times import load_compas, weights_at, MSSFR_FIDUCIAL, Z_SUN, CLASSES

HERE = Path(__file__).resolve().parent
COMPAS_DIR = HERE / 'data' / '10019887' / 'Output'


def main(outfile=HERE / 'processed_delay_times.h5'):
    compas = load_compas(COMPAS_DIR)
    tau = np.geomspace(TAU_MIN / 1e9, TAU_MAX / 1e9, 1000)     # delay times [Gyr]
    z_form = np.round(np.linspace(0, 10, 101), 2)               # formation redshifts

    with h5py.File(outfile, 'w') as f:
        f.attrs['description'] = ('Redshift-dependent COMPAS delay time distributions p(tau | z_form) of Schiebelbein-Zwack & Fishbach, '
                                  '"Forbidden Formation Histories: The Binary Black Hole Merger Rate Disfavours Long Delay Times". '
                                  'p(tau | z_form) = sum_k w_k(z_form) p_k(tau), k = low, mid, high metallicity (paper Eq. 6).')
        f.attrs['compas_source'] = 'Fishbach & van Son (2023), https://zenodo.org/records/10019887'
        f.attrs['metallicity_history'] = 'van Son et al. (2023) Eq. 11 skewed log-normal, Table 2 fiducial parameters'
        for k, v in MSSFR_FIDUCIAL.items():
            f.attrs[f'mssfr_{k}'] = v
        f.attrs['metallicity_classes'] = f'low: Z <= Zsun/10, mid: Zsun/10 < Z <= Zsun/2, high: Z > Zsun/2, Zsun = {Z_SUN}'
        f.attrs['normalisation'] = 'integral of p over tau_min <= tau <= tau_max is 1 at every z_form'

        d = f.create_dataset('tau', data=tau)
        d.attrs['units'] = 'Gyr'
        d.attrs['description'] = 'delay time, log-spaced from tau_min = 10 Myr to the age of the Universe (Planck 2015)'
        d = f.create_dataset('z_form', data=z_form)
        d.attrs['description'] = 'formation redshift'

        for name, model in compas.items():
            # class densities on the tau grid, each normalised between tau_min and tau_max
            # (a class without mergers stays zero and has zero weight)
            p_k = np.array([np.interp(tau, model['t'], p, left=0.0, right=0.0) for p in model['p']])
            area = trapezoid(p_k, tau, axis=1)
            p_k[area > 0] /= area[area > 0, None]
            w = weights_at(model, MSSFR_FIDUCIAL, z_form)                                    # (3, n_z)

            g = f.create_group(name)
            d = g.create_dataset('p_tau', data=(w.T @ p_k).astype(np.float32), compression='gzip', shuffle=True)
            d.attrs['units'] = 'Gyr^-1'
            d.attrs['shape'] = '(z_form, tau)'
            d = g.create_dataset('weights', data=w.T)
            d.attrs['description'] = 'fraction of BBH progenitors in each metallicity class, w_k(z_form)'
            d.attrs['shape'] = '(z_form, class)'
            d.attrs['classes'] = ','.join(CLASSES)
            d = g.create_dataset('p_tau_class', data=p_k.astype(np.float32), compression='gzip', shuffle=True)
            d.attrs['units'] = 'Gyr^-1'
            d.attrs['description'] = 'delay time distribution of each metallicity class, p_k(tau)'
            d.attrs['shape'] = '(class, tau)'

    print(f"wrote {len(compas)} models to {outfile}")


if __name__ == '__main__':
    main(*sys.argv[1:])
