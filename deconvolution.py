"""
Direct time-domain deconvolution of the BBH merger rate (paper Section 2.1).

The merger rate is the formation rate convolved with the delay time distribution (Eq. 1).
On a grid of cosmic time this is a lower-triangular linear system (Eq. 2), solved exactly
by forward substitution (Eq. 3).
"""
import numpy as np
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo   # Planck 2015, as in the LVK GWTC-4.0 population analyses
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.linalg import solve_triangular, toeplitz
from scipy.signal import windows
from scipy.stats import linregress

TAU_MIN = 1e7                               # minimum delay time: 10 Myr [yr]
TAU_MAX = cosmo.age(0).to_value(u.yr)       # maximum delay time: the age of the Universe [yr]


def z_from_age(t_yr):
    """
    Redshift at a given age of the Universe.

    Parameters:
        t_yr: array
            Age of the Universe [yr].
    Returns:
        z: array
            Redshift.
    """
    zr = np.linspace(0, 30, 6000)
    return np.interp(t_yr, cosmo.age(zr).to_value(u.yr)[::-1], zr[::-1])


class Grid:
    """
    Solver grid in cosmic time, from 10 Myr to the lowest-redshift point of the merger rate data.

    Parameters:
        times_data: array
            Age of the Universe at the merger rate data points [yr], decreasing (increasing redshift).
        n: int
            Number of grid steps.
        z_window: tuple
            Redshift range over which the formation rate must be non-negative.
        z_taper: float
            Redshift at which the merger rate reaches its boundary value.
    """

    def __init__(self, times_data, n=1000, z_window=(0.1, 1.5), z_taper=1.55):
        self.times_data = np.asarray(times_data, dtype=float)
        self.n = n
        self.T = np.linspace(TAU_MIN, self.times_data[0], n)
        self.dt = self.T[1] - self.T[0]
        self.z = z_from_age(self.T)
        t_lo, t_hi = cosmo.age(z_window[1]).to_value(u.yr), cosmo.age(z_window[0]).to_value(u.yr)
        self.window = (self.T > t_lo) & (self.T < t_hi)

        # half Tukey window joining the boundary value to the data between z = 1.5 and 1.55
        self.taper = (self.T >= cosmo.age(z_taper).to_value(u.yr)) & (self.T < self.times_data[-1])
        self.ramp = windows.tukey(2 * self.taper.sum(), alpha=1)[:self.taper.sum()]

        # delay-time bin edges of the kernel: tau_min + n dt
        self.edges = TAU_MIN + np.arange(n + 1) * self.dt

    def merger_rate(self, R_data, boundary=0.0):
        """
        Merger rate draw(s) on the grid.

        Parameters:
            R_data: array
                Merger rate at the data points, shape (n_data,) or (n_data, n_draws).
            boundary: float
                Merger rate beyond z = 1.55 [Gpc^-3 yr^-1].
        Returns:
            Rm: array
                Merger rate on the grid, shape (n,) or (n, n_draws).
        """
        R_data = np.asarray(R_data, dtype=float)
        Rm = np.full((self.n,) + R_data.shape[1:], float(boundary))
        inside = self.T >= self.times_data[-1]
        Rm[inside] = interp1d(self.times_data, R_data, axis=0)(self.T[inside])
        ext = interp1d(self.times_data, R_data, axis=0, bounds_error=False, fill_value='extrapolate')(self.T[self.taper])
        r = self.ramp.reshape((-1,) + (1,) * (R_data.ndim - 1))
        Rm[self.taper] = (1 - r) * boundary + r * ext
        return Rm

    def binned_kernel(self, t_gyr, p_gyr):
        """
        Delay time distribution binned on the grid's own steps (exact integral over each step),
        normalised between tau_min and tau_max.

        Parameters:
            t_gyr: array
                Delay times [Gyr].
            p_gyr: array
                Delay time distribution at t_gyr [Gyr^-1].
        Returns:
            kernel: array
                Mean probability density in each delay step [yr^-1] (zero if p_gyr is zero everywhere).
        """
        t = np.asarray(t_gyr, dtype=float) * 1e9
        P = cumulative_trapezoid(np.asarray(p_gyr, dtype=float) / 1e9, t, initial=0)
        F = np.interp(self.edges, t, P, left=0.0, right=P[-1])
        total = np.interp(TAU_MAX, t, P, right=P[-1]) - np.interp(TAU_MIN, t, P, left=0.0)
        if total <= 0:
            return np.zeros(len(self.edges) - 1)
        return np.diff(F) / total / self.dt

    def powerlaw_kernel(self, alpha):
        """
        Power-law delay time distribution p(tau) ∝ tau^alpha (Eq. 4), binned on the grid from its exact cumulative distribution.

        Parameters:
            alpha: float
                Power-law index.
        Returns:
            kernel: array
                Mean probability density in each delay step [yr^-1].
        """
        e = np.clip(self.edges, TAU_MIN, TAU_MAX)
        if np.isclose(alpha, -1):
            F = np.log(e / TAU_MIN) / np.log(TAU_MAX / TAU_MIN)
        else:
            F = (e**(alpha + 1) - TAU_MIN**(alpha + 1)) / (TAU_MAX**(alpha + 1) - TAU_MIN**(alpha + 1))
        return np.diff(F) / self.dt

    def forward_matrix(self, kernel, weights=None):
        """
        Convolution matrix K of Eq. 2, K[i, j] = probability that a progenitor formed in step j merges in step i.

        Parameters:
            kernel: array
                Binned kernel, shape (n,), or one kernel per metallicity class, shape (n_class, n).
            weights: array
                Fraction of progenitors in each class at each formation step, shape (n_class, n).
                Only used with class kernels.
        Returns:
            K: array
                Lower-triangular matrix, shape (n, n).
        """
        if weights is None:
            k = np.asarray(kernel, dtype=float) * self.dt
            return toeplitz(k, np.r_[k[0], np.zeros(self.n - 1)])
        return sum(self.forward_matrix(kc) * w[None, :] for kc, w in zip(kernel, weights))

    def is_physical(self, Rf):
        """
        True for each formation rate (column) that is non-negative everywhere in the positivity window.
        """
        return ~np.any(Rf[self.window] < 0, axis=0)

    def compatibility(self, K, Rm):
        """
        Fraction of merger rate draws whose formation rate is physical (Eq. 8).
        """
        return self.is_physical(solve(K, Rm)).mean()


def solve(K, Rm):
    """
    Formation rate(s) from merger rate(s), the exact solution of K R_form = R_merge (Eq. 3).

    Parameters:
        K: array
            Lower-triangular convolution matrix.
        Rm: array
            Merger rate(s) on the grid, one per column.
    Returns:
        Rf: array
            Formation rate(s) on the grid.
    """
    return solve_triangular(K, Rm, lower=True, check_finite=False)


def fit_slope(t, p):
    """
    Power-law index of a delay time distribution: log-log linear fit over tau <= 10 Gyr where p > 1e-6.

    Parameters:
        t: array
            Delay times [Gyr].
        p: array
            Delay time distribution [Gyr^-1].
    Returns:
        alpha: float
            Fitted power-law index.
    """
    mask = (t > 0) & (t <= 10) & (p > 1e-6)
    return linregress(np.log10(t[mask]), np.log10(p[mask])).slope


def alpha_at(alphas, P, threshold):
    """
    Power-law index where the compatibility P (falling with alpha) crosses a threshold, interpolated linearly.

    Parameters:
        alphas: array
            Increasing grid of power-law indices.
        P: array
            Compatibility at each alpha.
        threshold: float
            Compatibility threshold.
    Returns:
        alpha: float
            Crossing point (nan if P does not cross the threshold on the grid).
    """
    k = np.where(P < threshold)[0]
    if P[0] < threshold or len(k) == 0:
        return np.nan
    k = k[0]
    return alphas[k - 1] + (threshold - P[k - 1]) * (alphas[k] - alphas[k - 1]) / (P[k] - P[k - 1])


def weighted_percentiles(data, weights, percentiles=(5, 50, 95)):
    """
    Percentiles of weighted samples.

    Parameters:
        data: array
            Samples.
        weights: array
            Weight of each sample.
        percentiles: list
            Percentiles to compute.
    Returns:
        values: array
            Weighted percentiles.
    """
    ix = np.argsort(data)
    cw = np.cumsum(weights[ix])
    return np.interp(np.array(percentiles) / 100.0, cw / cw[-1], data[ix])
