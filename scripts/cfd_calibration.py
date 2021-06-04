# _*_ coding: utf-8 _*_

""" Perform the CFD calibration method.
refer to:
  Zhu, Y. and Y. Luo, 2015: Precipitation Calibration Based on the
  Frequency-Matching Method. Wea. Forecasting, 30, 1109–1124,
  https://doi.org/10.1175/WAF-D-13-00049.1
"""

import numpy as np


def cum_freq(data, bins=None, norm=True):
    """Calculate the cumulative frequency distribution.

    Arguments:
        data {numpy nd-array} -- numpy nd-array. The missing value is allowed.

    Keyword Arguments:
        bins {numpy array} -- the bin-edegs used to calculate CFD.
        norm {bool} -- normalize the distribution (default: {True})
    """

    # set the bin edges
    if bins is None:
        bins = np.concatenate(([0.1, 1], np.arange(2, 10, 1),
                               np.arange(10, 152, 2)))

    # mask the missing values and change negative to zero
    data = data[np.isfinite(data)]

    # calculate the cumulative frequency distribution
    cfd_array = np.full(bins.size, np.nan)
    for ib, b in enumerate(bins):
        cfd_array[ib] = np.count_nonzero(data >= b) * 1.0 / data.size

    # return the bin edges and CFD
    return cfd_array, bins


def cfd_match(tp, bins, obs_cfd, fcst_cfd):
    """Performe the frequency-matching methods.

    Arguments:
        tp {numpy array} -- model total precipitation forecast.
        bins {numpy array} -- bin edges of CFD.
        obs_cfd {numpy array} -- the observation CFD of training dataset.
        fcst_cfd {numpy array} -- the forecast CFD of training dataset.
    """

    # construct interpolation
    tp_cfd = np.interp(fcst_cfd, bins, tp.reval())
    return np.interp(bins, obs_cfd, tp_cfd)
