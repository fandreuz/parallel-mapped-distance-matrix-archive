from .parallel_futures import (
    fill_bins,
    compute_bounds,
    compute_padded_bounds,
    match_points_and_bins,
)


def compute_mapped_distance_on_chunk(
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
    samples1,
    samples2,
    max_distance,
    func,
    client,
    bins_per_axis=None,
    should_vectorize=True,
    exact_max_distance=True,
    bins_per_future=5,
):
    region_dimension = np.max(samples2, axis=0) - np.min(samples2, axis=0)

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
        samples1,
        bins_per_axis,
        region_dimension,
        bins_per_future=bins_per_future,
        client=client,
    )

    samples2_fu = client.scatter(samples2, broadcast=True)

    bins_bounds = client.map(compute_bounds, bins)
    # make this an iterable list
    samples2_fu = np.repeat(samples2_fu, len(bins_bounds))

    padded_bin_bounds = client.map(
        compute_padded_bounds, bins_bounds, max_distance=max_distance
    )
    inclusion_matrix = client.map(
        match_points_and_bins, padded_bin_bounds, samples2_fu
    )

    mapped_distance = np.zeros(
        (len(samples1), len(samples2)), dtype=samples1.dtype
    )
    results = client.map(
        compute_mapped_distance_on_chunk,
        bins,
        subgroups,
        inclusion_matrix,
        samples2_fu,
        max_distance=max_distance,
        exact_max_distance=exact_max_distance,
        function=func,
        out=mapped_distance,
    )
    client.gather(results)

    return mapped_distance
