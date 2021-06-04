# _*_ coding: utf-8 _*_

"""
Functions for computing probability-matched mean field.
"""

import numpy as np
from numba import jit


def prob_matched_mean(field, axis=0):
    """
    Calculate probability-matched ensemble mean.

    http://www.cawcr.gov.au/staff/eee/etrap/probmatch.html

    ------------------- How to do probability matching: ---------------------
      1. Rank the gridded rainfall from all n QPFs from largest to smallest,
         the keep every nth value starting with the n/2-th value.
      2. Rank the gridded rainfall from the ensemble mean from largest to
         smallest.
      3. Match the two histograms, mapping rain rates from (1) onto locations
         from (2).

    :param field: ensemble forecast field, multiple dimensional array
                  [..., ens, ..., lat, lon].
    :param axis: the ensemble dimension axis.
    :return: probability-matched ensemble mean array, [..., ..., lat, lon]
    """

    # check dimensions
    if field.ndim < 3:
        raise Exception("Input field should be array"
                        "[..., ens, ..., lat, lon].")

    # compute ensemble mean
    mean_field = np.mean(field, axis=axis)

    # move the ensemble dimension to first axis
    if axis != 0:
        dim = np.arange(field.ndim)
        dim[0] = dim[axis]
        dim[axis] = 0
        p_field = np.transpose(field, dim)
    else:
        p_field = field

    # get the shape of p_field and define the extra dimension
    shape = p_field.shape
    n_memb = shape[0]
    n_grid = shape[-2] * shape[-1]
    if len(shape) > 3:
        extra_dim = np.product(shape[1:-2])
    else:
        extra_dim = 1

    # reshape multiple dimensional field to 3D array
    p_field = np.reshape(p_field, (n_memb, extra_dim, n_grid))
    m_field = np.reshape(mean_field, (extra_dim, n_grid))

    # define probability matched result
    p_data = np.full((extra_dim, n_grid), np.nan)

    # loop every extra dimension
    for i in range(extra_dim):
        temp = p_field[:, i, :].flatten()
        p_data[i, :][np.argsort(m_field[i, :])] = temp[
            np.argsort(temp)[round(n_memb/2.0 - 1):(n_memb*n_grid):n_memb]]

    # reshape and return results
    p_data.shape = shape[1:]
    return p_data


@jit
def probability_matched_local(field1, field2, width=10):
    """
    Calculate local probability-matched field.

    Arguments:
        field1 {numpy 2d array} -- provides the spatial distribution.
        field2 {numpy 2d array} -- provides the quantity.

    Keyword Arguments:
        width {int} -- local square half width(grid number).(default: {10})

    Returns:
        numpy 2d array -- local probability-matched field.
    """

    # mask invalid array
    field_mask = np.logical_and(np.isfinite(field1), np.isfinite(field2))

    # define local probability-matched field
    shape = field1.shape
    field = np.full(shape, np.nan)

    # loop each grid point
    for j in range(shape[0]):
        for i in range(shape[1]):
            # invalid grid point
            if not field_mask[j, i]:
                continue

            # subset the loacl square
            i0 = max(0, i - width)
            i1 = min(shape[1], i + width + 1)
            j0 = max(0, j - width)
            j1 = min(shape[0], j + width + 1)
            sub1 = field1[j0:j1, i0:i1].copy()
            sub2 = field2[j0:j1, i0:i1]
            sub_mask = field_mask[j0:j1, i0:i1]
            sub3 = sub1[sub_mask]
            if sub3.size == 0:
                continue

            # perform probability-matching
            sub3[np.argsort(sub3)] = np.sort(sub2[sub_mask])
            sub1[sub_mask] = sub3
            field[j, i] = sub1[min(j, width), min(i, width)]

    # return field
    return field
