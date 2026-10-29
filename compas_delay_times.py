"""
Redshift-dependent COMPAS delay time distributions (paper Section 2.2, Eqs. 5-7).

The COMPAS release of Fishbach & van Son (2023), https://zenodo.org/records/10019887, gives delay time
histograms in three metallicity classes and the BBH yield in 14 metallicity bins. Combining them with the
metallicity distribution of star formation of van Son et al. (2023) gives p(tau | t_form) = sum_k w_k(t_form) p_k(tau).
"""
import os
import h5py
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.stats import norm

Z_SUN = 0.014                                           # solar metallicity of the release's class cuts
CLASS_EDGES = np.log([Z_SUN / 10, Z_SUN / 2])           # class boundaries in ln Z: low | mid | high
CLASSES = ('low', 'mid', 'high')
CLASS_COLUMNS = ('lowZ', 'midZ', 'highZ')               # column prefixes of the classes in the histogram files

# van Son et al. (2023) Table 2 fiducial parameters
MSSFR_FIDUCIAL = dict(mu0=0.025, muz=-0.049, alpha=-1.778, w0=1.125, wz=0.048)

Z_NODES = np.linspace(0, 10, 80)   # van Son et al. (2023) fit z = 0-10; the mix is held at its z = 10 value beyond


def model_name(filename):
    """COMPAS model name from a histogram file name."""
    return filename[len('histogram_data_'):-len('_LOG.csv')]


def density_per_tau(log10_t, dN_dlog10t, n_grid=100000):
    """
    Convert a histogram per log10 tau to a probability density per tau, interpolated linearly between bin centres.

    Parameters:
        log10_t: array
            Bin centres, log10(tau / Gyr).
        dN_dlog10t: array
            Number of mergers per unit log10 tau.
        n_grid: int
            Number of points of the output grid.
    Returns:
        t: array
            Delay times [Gyr], linear grid between the first and last bin centre.
        p: array
            Probability density [Gyr^-1], unit area over t (zero if the histogram is empty).
    """
    t_raw = 10**log10_t
    f = interp1d(t_raw, dN_dlog10t / (t_raw * np.log(10)), bounds_error=False, kind='linear', fill_value=0.0)
    t = np.linspace(t_raw.min(), t_raw.max(), n_grid)
    p = f(t)
    area = trapezoid(p, t)
    return t, (p / area if area > 0 else p)


def load_compas(directory):
    """
    Read the delay time distributions and yields of every COMPAS model with per-metallicity yields.

    Parameters:
        directory: str
            The release's Output folder.
    Returns:
        models: dict
            Model name -> dict with
            't': delay times [Gyr];
            'p': delay time distribution of each metallicity class, shape (3, len(t)) [Gyr^-1];
            'has_mergers': whether each class has any mergers in the histograms, shape (3,);
            'logZ': log10 metallicity of the 14 yield bins;
            'eta': BBH yield of each bin [Msun^-1].
    """
    files = sorted(f for f in os.listdir(directory) if f.startswith('histogram_data_') and f.endswith('_LOG.csv'))
    models = {}
    with h5py.File(os.path.join(directory, 'Yield_data.h5'), 'r') as yields:
        for f in files:
            name = model_name(f)
            if name not in yields:          # fWR1.0coolWind1.0 has no per-metallicity yields
                continue
            path = os.path.join(directory, f)
            with open(path) as fh:
                columns = fh.readline().lstrip('#').strip().split(',')
            data = np.loadtxt(path, delimiter=',', skiprows=1)
            curves = [density_per_tau(data[:, columns.index(f'{c}_xvalues')], data[:, columns.index(f'{c}_yvalues')])
                      for c in CLASS_COLUMNS]
            t = curves[0][0]
            p = np.vstack([np.interp(t, tc, pc) for tc, pc in curves])
            rec = yields[name][:]
            eta = rec['merging_BBH'] / rec['Average_SF_mass_needed'] / rec['mixture_weight_sum']
            models[name] = dict(t=t, p=p, has_mergers=p.any(axis=1), logZ=rec['Metallicity_bin'], eta=eta)
    return models


def dP_dlnZ(lnZ, z, p):
    """
    van Son et al. (2023) skewed log-normal metallicity distribution of star formation (Eq. 7), per unit ln Z.

    Parameters:
        lnZ: array
            Natural log of metallicity.
        z: array
            Redshift (broadcast against lnZ).
        p: dict
            Parameters mu0, muz, alpha, w0, wz.
    Returns:
        dP_dlnZ: array
            Probability density per unit ln Z.
    """
    w = p['w0'] * 10**(p['wz'] * z)
    delta = p['alpha'] / np.sqrt(1 + p['alpha']**2)
    xi = np.log(p['mu0'] * 10**(p['muz'] * z) / (2 * norm.cdf(delta * w))) - w**2 / 2
    x = (lnZ - xi) / w
    return 2 / w * norm.pdf(x) * norm.cdf(p['alpha'] * x)


def star_formation_per_bin(logZ, zs, p, n=41):
    """
    Fraction of star formation in the part of each COMPAS metallicity bin that lies in each class, S_m(z).
    The two bins that straddle a class boundary are split there.

    Parameters:
        logZ: array
            log10 metallicity of the bin centres (bins evenly spaced in log10 Z).
        zs: array
            Redshifts.
        p: dict
            van Son et al. (2023) parameters.
        n: int
            Integration points per bin.
    Returns:
        S: array
            Shape (len(zs), n_bins, 3).
    """
    half = np.log(10) * np.diff(logZ).mean() / 2
    lo, hi = logZ * np.log(10) - half, logZ * np.log(10) + half
    cuts = np.r_[-np.inf, CLASS_EDGES, np.inf]
    out = []
    for k in range(3):
        a, b = np.clip(lo, cuts[k], cuts[k + 1]), np.clip(hi, cuts[k], cuts[k + 1])
        l = np.linspace(a, b, n).T
        out.append(trapezoid(dP_dlnZ(l[None], np.atleast_1d(zs)[:, None, None], p), l[None], axis=-1))
    return np.stack(out, axis=-1)


def progenitor_weights(model, p, zs):
    """
    Fraction of BBH progenitors forming at each redshift in the low/mid/high metallicity class, w_k (Eq. 6).
    A class with no mergers in the delay time histograms gets zero weight.

    Parameters:
        model: dict
            COMPAS model from load_compas.
        p: dict
            van Son et al. (2023) parameters.
        zs: array
            Formation redshifts.
    Returns:
        w: array
            Shape (len(zs), 3).
    """
    w = np.einsum('i,zik->zk', model['eta'], star_formation_per_bin(model['logZ'], zs, p)) * model['has_mergers']
    return w / w.sum(axis=1, keepdims=True)


def weights_at(model, p, zs):
    """
    w_k at arbitrary formation redshifts, interpolated from Z_NODES (constant beyond z = 10).

    Returns:
        w: array
            Shape (3, len(zs)).
    """
    wn = progenitor_weights(model, p, Z_NODES)
    return np.array([np.interp(zs, Z_NODES, wn[:, k]) for k in range(3)])


def delay_time_distribution(model, w):
    """
    p(tau | t_form) = sum_k w_k p_k(tau) for one formation time (Eq. 6).

    Parameters:
        model: dict
            COMPAS model from load_compas.
        w: array
            Class fractions, shape (3,).
    Returns:
        p: array
            Delay time distribution on model['t'] [Gyr^-1].
    """
    return w @ model['p']
