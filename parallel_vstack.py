from serial import group_by

import numpy as np
import dask.array as da


def group_by(a):
    a = a[a[:, 0].argsort()]
    return np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:])


def fill_bins(pts, bins_per_axis, region_dimension, bins_array_chunks):
    h = np.divide(region_dimension, bins_per_axis)

    bin_coords = np.floor(np.divide(pts, h)).astype(int)
    # moves to the last bin of the axis any point which is outside the region
    # defined by pts2.
    bin_coords = np.clip(bin_coords, None, bins_per_axis - 1)

    # for each non-uniform point, gives the linearized coordinate of the
    # appropriate bin
    shifted_nbins_per_axis = np.ones_like(bins_per_axis)
    shifted_nbins_per_axis[:-1] = bins_per_axis[1:]
    linearized_bin_coords = np.sum(bin_coords * shifted_nbins_per_axis, axis=1)

    # add a second column containing the index in pts1
    linearized_bin_coords = np.hstack(
        [linearized_bin_coords[:, None], np.arange(len(pts))[:, None]]
    )
    indexes_inside_bins = group_by(linearized_bin_coords)

    nbins = len(indexes_inside_bins)
    lengths = tuple(map(len, indexes_inside_bins))
    biggest_bin = max(lengths)
    smallest_bin = min(lengths)

    if bins_array_chunks == "smallest-bin":
        bins_array_chunks = (1, smallest_bin, pts.shape[1])

    bins = da.from_array(
        np.zeros((nbins, biggest_bin, pts.shape[1])),
        chunks=bins_array_chunks,
    )
    for bin_idx in range(nbins):
        ps = pts[indexes_inside_bins[bin_idx]]
        bins[bin_idx, : len(ps)] = ps

    return bins, indexes_inside_bins


def compute_bounds(bins):
    bounds = da.max(
        bins[:, None, :] * np.array([-1, 1])[:, None, None], axis=2
    )
    bounds[:, 0] *= -1
    return bounds


def compute_padded_bounds(boundaries, distance):
    return boundaries - (distance * np.array([1, -1]))[None, :, None]


def match_points_and_bins(bins_bounds, points):
    broadcastable_points = points[:, None]
    return da.logical_and(
        da.all(bins_bounds[:, 0] < broadcastable_points, axis=2),
        da.all(broadcastable_points < bins_bounds[:, 1], axis=2),
    )


def compute_distance(pts1, pts2):
    return da.linalg.norm(pts1[:, None, :] - pts2[None, ...], axis=-1)


def compute_mapped_distance_on_bin(
    pts1_in_bin,
    pts2,
    inclusion_vector,
    max_distance,
    func,
    exact_max_distance,
    submatrix_mapped_distance_chunks,
):
    padded_bin_pts2 = pts2[inclusion_vector]
    bin_pts2_indexing_to_full = np.arange(len(pts2))[inclusion_vector]

    distances_da = compute_distance(pts1_in_bin, padded_bin_pts2)
    mapped_distance = da.map_blocks(func, distances_da, dtype=np.float64)

    if exact_max_distance:
        distances = distances_da.compute()
        too_far = distances > max_distance
        mapped_distance = mapped_distance.compute()
        mapped_distance[too_far] = 0

    submatrix = da.from_array(
        np.zeros((len(pts1_in_bin), len(pts2))),
        chunks=submatrix_mapped_distance_chunks,
    )
    submatrix[:, bin_pts2_indexing_to_full] = da.squeeze(mapped_distance)
    return submatrix


def mapped_distance_matrix(
    pts1,
    pts2,
    max_distance,
    func,
    bins_per_axis=None,
    should_vectorize=True,
    exact_max_distance=True,
    chunks="auto",
    bins_array_chunks="smallest-bin",
    submatrix_chunks="auto",
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
        pts1,
        bins_per_axis,
        region_dimension,
        bins_array_chunks=bins_array_chunks,
    )
    bins_bounds = compute_bounds(bins)
    padded_bin_bounds = compute_padded_bounds(bins_bounds, max_distance)

    inclusion_matrix = match_points_and_bins(padded_bin_bounds, pts2)

    # we exclude some bins due to the fact that no points in pts2 belong to them

    # boolean indexing with dask arrays does not work at the moment
    # non_empty_bins = da.any(inclusion_matrix, axis=0)

    inclusion_matrix = inclusion_matrix.compute()
    non_empty_bins = np.any(inclusion_matrix, axis=0)

    bins = bins[non_empty_bins]
    inclusion_matrix = inclusion_matrix[:, non_empty_bins].T

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

    mapped_distance = da.from_array(
        np.zeros((len(pts1), len(pts2))), chunks=chunks
    )
    mapped_distance[non_empty_bins_mapping] = da.vstack(
        [
            compute_mapped_distance_on_bin(
                bin[:n_in_bin],
                pts2,
                inclusion_vector,
                max_distance=max_distance,
                func=func,
                exact_max_distance=exact_max_distance,
                submatrix_mapped_distance_chunks=submatrix_chunks,
            )
            for bin, n_in_bin, inclusion_vector in zip(
                bins, map(len, indexes_inside_bins), inclusion_matrix
            )
        ]
    )

    return mapped_distance
