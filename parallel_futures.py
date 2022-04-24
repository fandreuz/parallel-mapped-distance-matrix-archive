import numpy as np

from dask.distributed import as_completed


def group_by(a):
    a = a[a[:, 0].argsort()]
    return np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:])


def bins_from_idxes(pre_bins, idxes_in_bins, idx):
    nbins = len(idxes_in_bins)
    bin_sizes = np.fromiter(map(len, idxes_in_bins), dtype=int)
    bin_starts = np.concatenate([[0], np.cumsum(bin_sizes[:-1])])
    biggest_bin = np.max(bin_sizes)

    bins = np.zeros(
        (nbins, biggest_bin, pre_bins.shape[1]), dtype=pre_bins.dtype
    )
    for bin_idx in range(nbins):
        bin_size = bin_sizes[bin_idx]
        start = bin_starts[bin_idx]
        bins[bin_idx, :bin_size] = pre_bins[start : start + bin_size]
    return bins, idx


def fill_bins(pts, bins_per_axis, region_dimension, bins_per_future, client):
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
    indexes_inside_bins = list(map(np.array, indexes_inside_bins))
    nbins = len(indexes_inside_bins)

    n_subgroups = nbins // bins_per_future
    n_subgroups += 1 if nbins % bins_per_future != 0 else 0

    subgroups = tuple(
        indexes_inside_bins[i * bins_per_future : (i + 1) * bins_per_future]
        for i in range(n_subgroups)
    )
    flattened_subgroups = tuple(map(np.concatenate, subgroups))

    pre_bins = client.scatter([pts[fsg] for fsg in flattened_subgroups])
    bins = client.map(bins_from_idxes, pre_bins, subgroups, range(n_subgroups))

    return bins, subgroups, flattened_subgroups


def compute_bounds(bins):
    # this contains also a pointer to the subgroup idx
    bins, _ = bins

    plus_minus = np.array([-1, 1], dtype=int)[:, None]
    # find maximum of positive and negative axes
    bounds = np.max(bins[:, :, None] * plus_minus, axis=1)
    bounds[:, 0] *= -1
    return bounds


def compute_padded_bounds(boundaries, max_distance):
    plus_minus = max_distance * np.array([-1, 1], dtype=int)[:, None]
    return boundaries + plus_minus


def match_points_and_bins(bins_bounds, points):
    broadcastable_points = points[None]
    return np.logical_and(
        np.all(bins_bounds[:, None, 0] < broadcastable_points, axis=2),
        np.all(broadcastable_points < bins_bounds[:, None, 1], axis=2),
    )


def compute_mapped_distance_on_chunk(
    pts1_in_future,
    subgroups,
    inclusion_submatrix,
    pts2,
    max_distance,
    function,
    exact_max_distance,
):
    # this contains also a pointer to the subgroup idx, as well as the set of
    # points in pts1 which belong to this future
    pts1_in_future, idx = pts1_in_future

    bins_sizes = np.fromiter(map(len, subgroups), dtype=int)
    bin_start_indexes = np.concatenate(
        ([0], np.cumsum(bins_sizes[:-1])), dtype=int
    )
    non_trivial_bins = np.arange(len(pts1_in_future))[
        np.logical_and(bins_sizes > 0, np.any(inclusion_submatrix, axis=1))
    ]

    submatrix = np.zeros(
        (np.sum(bins_sizes), len(pts2)), dtype=pts1_in_future.dtype
    )

    for bin_idx in non_trivial_bins:
        bin_size = bins_sizes[bin_idx]
        pts1_in_bin = pts1_in_future[bin_idx, :bin_size]

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
        end = start + bin_size
        submatrix[
            start:end,
            inclusion_vector,
        ] = mapped_distance
    return submatrix, idx


def shared_compute_mapped_distance_on_chunk(
    pts1_in_future,
    subgroups,
    inclusion_submatrix,
    pts2,
    max_distance,
    function,
    exact_max_distance,
    out=None,
):
    # this contains also a pointer to the subgroup idx, as well as the set of
    # points in pts1 which belong to this future
    pts1_in_future, idx = pts1_in_future

    bins_sizes = np.fromiter(map(len, subgroups), dtype=int)
    non_trivial_bins = np.arange(len(pts1_in_future))[
        np.logical_and(bins_sizes > 0, np.any(inclusion_submatrix, axis=1))
    ]

    for bin_idx in non_trivial_bins:
        bin_size = bins_sizes[bin_idx]
        pts1_in_bin = pts1_in_future[bin_idx, :bin_size]

        inclusion_vector_idxes = np.where(inclusion_submatrix[bin_idx])[0]
        padded_bin_pts2 = pts2[inclusion_vector_idxes]

        distances = np.linalg.norm(
            pts1_in_bin[:, None, :] - padded_bin_pts2[None, ...], axis=-1
        )

        if exact_max_distance:
            nearby = distances < max_distance
            mapped_distance = np.zeros_like(distances, dtype=pts1_in_bin.dtype)
            mapped_distance[nearby] = function(distances[nearby])
        else:
            mapped_distance = function(distances)

        out[
            subgroups[bin_idx][:, None], inclusion_vector_idxes[None]
        ] = mapped_distance


def mapped_distance_matrix(
    pts1,
    pts2,
    max_distance,
    func,
    client,
    bins_per_axis=None,
    should_vectorize=True,
    exact_max_distance=True,
    bins_per_future=5,
    shared_memory=False,
):
    region_dimension = np.max(pts2, axis=0) - np.min(pts2, axis=0)

    if bins_per_future <= 1:
        raise ValueError("At least two bins per Future.")

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

    bins, subgroups, flattened_subgroups = fill_bins(
        pts1,
        bins_per_axis,
        region_dimension,
        bins_per_future=bins_per_future,
        client=client,
    )

    pts2_fu = client.scatter(pts2, broadcast=True)

    bins_bounds = client.map(compute_bounds, bins)
    # make this an iterable list
    pts2_fu = np.repeat(pts2_fu, len(bins_bounds))

    padded_bin_bounds = client.map(
        compute_padded_bounds, bins_bounds, max_distance=max_distance
    )
    inclusion_matrix = client.map(
        match_points_and_bins, padded_bin_bounds, pts2_fu
    )

    mapped_distance = np.zeros((len(pts1), len(pts2)), dtype=pts1.dtype)
    if not shared_memory:
        # all the writes to the global distance matrix occur in the main
        # thread
        mapped_distances_fu = client.map(
            compute_mapped_distance_on_chunk,
            bins,
            subgroups,
            inclusion_matrix,
            pts2_fu,
            max_distance=max_distance,
            exact_max_distance=exact_max_distance,
            function=func,
        )

        for _, (submatrix, idx) in as_completed(
            mapped_distances_fu, with_results=True
        ):
            mapped_distance[flattened_subgroups[idx]] = submatrix
    else:
        results = client.map(
            shared_compute_mapped_distance_on_chunk,
            bins,
            subgroups,
            inclusion_matrix,
            pts2_fu,
            max_distance=max_distance,
            exact_max_distance=exact_max_distance,
            function=func,
            out=mapped_distance,
        )
        client.gather(results)

    return mapped_distance
