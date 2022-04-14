from serial import group_by

import numpy as np
import dask.array as da
from numbers import Integral, Number
from dask.array.core import Array

from functools import partial


def group_by(a):
    a = a[a[:, 0].argsort()]
    return np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:])


def fill_bins(pts, bins_per_axis, region_dimension, bins_per_chunk):
    h = np.divide(region_dimension, bins_per_axis)

    pts_da = da.from_array(pts, chunks=("auto", -1))
    bin_coords = da.floor_divide(pts, h).astype(int)
    # moves to the last bin of the axis any point which is outside the region
    # defined by pts2.
    da.clip(bin_coords, None, bins_per_axis - 1, out=bin_coords)

    # for each non-uniform point, gives the linearized coordinate of the
    # appropriate bin
    shifted_nbins_per_axis = np.ones_like(bins_per_axis)
    shifted_nbins_per_axis[:-1] = bins_per_axis[1:]
    # we re-use the first column of bin_coords
    linearized_bin_coords = da.dot(bin_coords, shifted_nbins_per_axis[:, None])
    linearized_bin_coords = da.hstack(
        [linearized_bin_coords, da.arange(len(pts))[:, None]]
    ).compute()

    indexes_inside_bins = group_by(linearized_bin_coords)

    nbins = len(indexes_inside_bins)
    lengths = tuple(map(len, indexes_inside_bins))
    biggest_bin = max(lengths)
    smallest_bin = min(lengths)

    bins = da.from_array(
        np.zeros((nbins, biggest_bin, pts.shape[1])),
        chunks=(bins_per_chunk, -1, -1),
    )
    # TODO improve this
    for bin_idx in range(nbins):
        ps = pts[indexes_inside_bins[bin_idx]]
        bins[bin_idx, : len(ps)] = ps

    return bins, indexes_inside_bins


def compute_bounds(bins):
    plus_minus = np.array([-1, 1])[:, None]
    nbins_per_chunk = max(bins.chunks[0])
    _, npts_per_bin, ndims = bins.shape

    # we add a negative axis
    temp = da.map_blocks(
        lambda bns: bns[:, :, None] * plus_minus,
        bins,
        dtype=bins.dtype,
        chunks=(nbins_per_chunk, npts_per_bin, 2, ndims),
        new_axis=(2,),
        meta=np.array((), dtype=bins.dtype),
    )

    # find maximum of positive and negative axes
    bounds = da.max(temp, axis=1)

    def switch_sign(x):
        x[:, 0] *= -1
        return x

    # we restore the original sign of the negative axis
    return da.map_blocks(
        switch_sign,
        bounds,
        dtype=bounds.dtype,
        meta=np.array((), dtype=bins.dtype),
    )


def compute_padded_bounds(boundaries, distance):
    plus_minus = distance * np.array([-1, 1])[:, None]
    return da.map_blocks(
        lambda bs: bs + plus_minus,
        boundaries,
        meta=np.array((), dtype=boundaries.dtype),
    )


def match_points_and_bins(bins_bounds, points):
    broadcastable_points = points[:, None]
    return da.logical_and(
        da.all(bins_bounds[:, 0] < broadcastable_points, axis=2),
        da.all(broadcastable_points < bins_bounds[:, 1], axis=2),
    )


def compute_mapped_distance_on_chunk(
    pts1_in_chunk,
    n_pts1_inside_chunk,
    inclusion_submatrix,
    pts2,
    max_distance,
    func,
    exact_max_distance,
):
    # re-establish the proper dimensions for these matrices
    n_pts1_inside_chunk = n_pts1_inside_chunk[:,0,0]
    inclusion_submatrix = inclusion_submatrix[...,0]

    # TODO shall this be a dask array?
    submatrix = np.zeros((np.sum(n_pts1_inside_chunk), len(pts2)))

    last_written = 0
    for bin_idx in range(len(pts1_in_chunk)):
        pts1_in_bin = pts1_in_chunk[bin_idx, : n_pts1_inside_chunk[bin_idx]]

        inclusion_vector = inclusion_submatrix[bin_idx]
        padded_bin_pts2 = pts2[inclusion_vector]

        distances_da = da.linalg.norm(
            pts1_in_bin[:, None, :] - padded_bin_pts2[None, ...], axis=-1
        )
        mapped_distance = func(distances_da)

        if exact_max_distance:
            distances = distances_da
            too_far = distances > max_distance
            mapped_distance = mapped_distance
            mapped_distance[too_far] = 0

        submatrix[
            last_written : last_written + n_pts1_inside_chunk[bin_idx],
            inclusion_vector,
        ] = mapped_distance
        last_written += n_pts1_inside_chunk[bin_idx]
    return submatrix


def mapped_distance_matrix(
    pts1,
    pts2,
    max_distance,
    func,
    bins_per_axis=None,
    should_vectorize=True,
    exact_max_distance=True,
    bins_per_thread="auto",
):
    region_dimension = np.max(pts2, axis=0) - np.min(pts2, axis=0)

    # not using np.vectorize if the function is already vectorized allows us
    # to save some time
    if should_vectorize:
        func = np.vectorize(func)

    if not bins_per_axis:
        # 3 bins per axis,
        # i.e. 9 bins in 2D
        bins_per_axis = np.full(region_dimension.shape, 3)
    else:
        bins_per_axis = np.asarray(bins_per_axis)

    if bins_per_axis.dtype != int:
        raise ValueError("The number of bins must be an integer number")

    bins, indexes_inside_bins = fill_bins(
        pts1, bins_per_axis, region_dimension, bins_per_chunk=bins_per_thread
    )
    bins_bounds = compute_bounds(bins)
    padded_bin_bounds = compute_padded_bounds(bins_bounds, max_distance)

    inclusion_matrix_da = match_points_and_bins(padded_bin_bounds, pts2).T

    # we exclude some bins due to the fact that no points in pts2 belong to them

    # boolean indexing with dask arrays does not work at the moment
    # non_empty_bins = da.any(inclusion_matrix, axis=0)

    inclusion_matrix = inclusion_matrix_da.compute()
    non_empty_bins = np.any(inclusion_matrix, axis=1)

    bins = bins[non_empty_bins]
    inclusion_matrix = inclusion_matrix[non_empty_bins]

    nbins = len(indexes_inside_bins)
    # this is used to move in the proper position of the final matrix
    # the row of the matrix produced by vstack
    indexes_inside_bins = [
        indexes_inside_bins[i] for i in range(nbins) if non_empty_bins[i]
    ]
    non_empty_bins_mapping = np.concatenate(
        indexes_inside_bins,
        dtype=int,
    )

    # we take one for each bin in each chunk
    n_pts1_inside_bins = np.fromiter(map(len, indexes_inside_bins), dtype=int)

    if isinstance(bins.chunks[0], int):
        new_chunks_pts1 = tuple(
            n_pts1_inside_bins.reshape(-1, bins.chunks[0]).sum(axis=1)
        )
    else:
        indexes = np.cumsum(np.fromiter(bins.chunks[0], dtype=int))
        # TODO this can be made more NumPy-thonic
        new_chunks_pts1 = tuple(
            map(
                sum,
                filter(
                    lambda l: len(l) > 0, np.split(n_pts1_inside_bins, indexes)
                ),
            )
        )

    pcompute_mapped_distance_on_chunk = partial(
        compute_mapped_distance_on_chunk,
        pts2=pts2,
        max_distance=max_distance,
        exact_max_distance=exact_max_distance,
        func=func,
    )

    n_pts1_inside_bins_da = da.from_array(
        n_pts1_inside_bins[:, None, None], chunks=(bins.chunks[0], 1, 1)
    )
    inclusion_matrix_da = inclusion_matrix_da[..., None].rechunk(
        (inclusion_matrix_da.chunks[0], inclusion_matrix_da.chunks[1], (1,))
    )

    mapped_distance_chunks = (new_chunks_pts1, (len(pts2),))
    mapped_distance = np.zeros((len(pts1), len(pts2)))
    mapped_distance[non_empty_bins_mapping] = da.map_blocks(
        pcompute_mapped_distance_on_chunk,
        bins,
        n_pts1_inside_bins_da,
        inclusion_matrix_da,
        # bins are aggregated (i.e. we lose the first dimension of bins)
        # and the spatial dimension is lost due to the fact that distance
        # is a scalar
        drop_axis=(1, 2),
        # the second dimension is the number of points in pts2
        new_axis=(1,),
        chunks=mapped_distance_chunks,
        dtype=pts1.dtype,
        meta=np.array((), dtype=pts1.dtype),
    ).compute()

    return mapped_distance
