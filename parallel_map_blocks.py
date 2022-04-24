import numpy as np
import dask.array as da


def group_by(a):
    a = a[a[:, 0].argsort()]
    return np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:])


def bins_from_idxes(idxes_in_bin, per_bin_size, pts, bins_size):
    per_bin_size = per_bin_size[:, 0]
    nbins = len(idxes_in_bin)
    bins = np.zeros((nbins, bins_size, pts.shape[1]), dtype=pts.dtype)
    for bin_idx in range(nbins):
        size = per_bin_size[bin_idx]
        bins[bin_idx, :size] = pts[idxes_in_bin[bin_idx, :size]]
    return bins


def fill_bins(pts, bins_per_axis, region_dimension, bins_per_chunk):
    h = np.divide(region_dimension, bins_per_axis)

    bin_coords = np.floor_divide(pts, h).astype(int)
    # moves to the last bin of the axis any point which is outside the region
    # defined by pts2.
    np.clip(bin_coords, None, bins_per_axis - 1, out=bin_coords)

    shifted_nbins_per_axis = np.ones_like(bins_per_axis, dtype=int)
    shifted_nbins_per_axis[:-1] = bins_per_axis[1:]
    # for each non-uniform point, this gives the linearized coordinate of the
    # appropriate bin
    linearized_bin_coords = np.dot(bin_coords, shifted_nbins_per_axis[:, None])
    aug_linearized_bin_coords = np.hstack(
        [linearized_bin_coords, np.arange(len(pts))[:, None]]
    )

    indexes_inside_bins = group_by(aug_linearized_bin_coords)

    nbins = len(indexes_inside_bins)
    biggest_bin = max(map(len, indexes_inside_bins))

    padded_indexes_inside_bins = np.full((nbins, biggest_bin), -1, dtype=int)
    per_bin_size = np.empty((nbins), dtype=int)
    for bin_idx in range(nbins):
        bin_content = indexes_inside_bins[bin_idx]
        lbc = len(bin_content)

        per_bin_size[bin_idx] = lbc
        padded_indexes_inside_bins[bin_idx, :lbc] = bin_content

    if bins_per_chunk == "auto":
        bins_chunks = ("auto", (biggest_bin,), (pts.shape[1],))
    else:
        bins_chunks = (bins_per_chunk, (biggest_bin,), (pts.shape[1],))

    padded_indexes_inside_bins = da.from_array(
        padded_indexes_inside_bins,
        chunks=(bins_per_chunk, -1),
    )

    da_per_bin_size = da.from_array(
        per_bin_size[:, None],
        chunks=(bins_per_chunk, (1,)),
    )

    bins = da.map_blocks(
        bins_from_idxes,
        padded_indexes_inside_bins,
        da_per_bin_size,
        pts=pts,
        bins_size=biggest_bin,
        chunks=(*padded_indexes_inside_bins.chunks, pts.shape[1]),
        meta=np.array((), dtype=pts.dtype),
        name="bins",
        new_axis=(2,),
        dtype=pts.dtype,
    )

    return bins, per_bin_size, indexes_inside_bins


def compute_bounds(bins):
    plus_minus = np.array([-1, 1], dtype=int)[:, None]
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
        name="2x!pts1",
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
        name="max_min_per_bin",
    )


def compute_padded_bounds(boundaries, distance):
    plus_minus = distance * np.array([-1, 1], dtype=int)[:, None]
    return da.map_blocks(
        lambda bs: bs + plus_minus,
        boundaries,
        meta=np.array((), dtype=boundaries.dtype),
        name="padded_bounds",
    )


def match_points_and_bins(bins_bounds, points):
    broadcastable_points = points[None]
    return da.logical_and(
        da.all(bins_bounds[:, None, 0] < broadcastable_points, axis=2),
        da.all(broadcastable_points < bins_bounds[:, None, 1], axis=2),
    )


def compute_mapped_distance_on_chunk(
    pts1_in_chunk,
    n_pts1_inside_chunk,
    inclusion_submatrix,
    pts2,
    max_distance,
    function,
    exact_max_distance,
):
    # re-establish the proper dimensions for these matrices
    n_pts1_inside_chunk = n_pts1_inside_chunk[:, 0, 0]
    inclusion_submatrix = inclusion_submatrix[..., 0]

    bin_start_indexes = np.concatenate(
        ([0], np.cumsum(n_pts1_inside_chunk[:-1])), dtype=int
    )

    # TODO shall this be a dask array?
    submatrix = np.zeros(
        (np.sum(n_pts1_inside_chunk), len(pts2)), dtype=pts1_in_chunk.dtype
    )

    non_trivial_bins = np.arange(len(pts1_in_chunk))[
        np.logical_and(
            n_pts1_inside_chunk > 0, np.any(inclusion_submatrix, axis=1)
        )
    ]

    for bin_idx in non_trivial_bins:
        pts1_in_bin = pts1_in_chunk[bin_idx, : n_pts1_inside_chunk[bin_idx]]

        inclusion_vector = inclusion_submatrix[bin_idx]
        padded_bin_pts2 = pts2[inclusion_vector]

        distances = np.linalg.norm(
            pts1_in_bin[:, None, :] - padded_bin_pts2[None, ...], axis=-1
        )

        if exact_max_distance:
            nearby = distances < max_distance
            mapped_distance = np.zeros_like(distances, dtype=pts1_in_bin.dtype)
            mapped_distance[nearby] = function(distances[nearby])
        else:
            mapped_distance = function(distances)

        start = bin_start_indexes[bin_idx]
        submatrix[
            start : start + n_pts1_inside_chunk[bin_idx],
            inclusion_vector,
        ] = mapped_distance
    return submatrix


def mapped_distance_matrix(
    pts1,
    pts2,
    max_distance,
    func,
    bins_per_axis=None,
    should_vectorize=True,
    exact_max_distance=True,
    bins_per_chunk="auto",
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

    bins, n_pts1_inside_bins, indexes_inside_bins = fill_bins(
        pts1,
        bins_per_axis,
        region_dimension,
        bins_per_chunk=bins_per_chunk,
    )
    bins_bounds = compute_bounds(bins)
    padded_bin_bounds = compute_padded_bounds(bins_bounds, max_distance)

    inclusion_matrix_da = match_points_and_bins(padded_bin_bounds, pts2)
    del bins_bounds
    del padded_bin_bounds

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

    n_pts1_inside_bins_da = da.from_array(
        n_pts1_inside_bins[:, None, None],
        chunks=(bins.chunks[0], 1, 1),
        name="n_pts1_per_bin",
        meta=np.array((), dtype=int),
    )
    inclusion_matrix_da = inclusion_matrix_da[..., None].rechunk(
        (inclusion_matrix_da.chunks[0], inclusion_matrix_da.chunks[1], (1,))
    )

    bins_mapping = np.concatenate(
        indexes_inside_bins,
        dtype=int,
    )

    mapped_distance_chunks = (new_chunks_pts1, (len(pts2),))
    mapped_distance = da.from_array(
        np.zeros((len(pts1), len(pts2)), dtype=pts1.dtype),
        name="mapped_distance",
        meta=np.array((), dtype=pts1.dtype),
    )

    mapped_distance[bins_mapping] = da.map_blocks(
        compute_mapped_distance_on_chunk,
        bins,
        n_pts1_inside_bins_da,
        inclusion_matrix_da,
        pts2=pts2,
        max_distance=max_distance,
        exact_max_distance=exact_max_distance,
        function=func,
        # bins are aggregated (i.e. we lose the first dimension of bins)
        # and the spatial dimension is lost due to the fact that distance
        # is a scalar
        drop_axis=(1, 2),
        # the second dimension is the number of points in pts2
        new_axis=(1,),
        chunks=mapped_distance_chunks,
        dtype=pts1.dtype,
        meta=np.array((), dtype=pts1.dtype),
        name="submt_mapped_distance",
    )

    return mapped_distance
