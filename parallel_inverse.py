import numpy as np

from dask.distributed import as_completed


def group_by(a):
    a = a[a[:, 0].argsort()]
    return np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:])


def fill_bins(
    uniform_grid_cell_size,
    uniform_grid_cell_count,
    pts,
    bins_size,
    pts_per_future,
    client,
):
    bin_coords = np.floor_divide(
        pts, uniform_grid_cell_size * bins_size
    ).astype(int)
    # moves to the last bin of the axis any point which is outside the region
    # defined by samples2.
    np.clip(bin_coords, None, uniform_grid_cell_count - 1, out=bin_coords)

    bins_per_axis = uniform_grid_cell_count // bins_size

    shifted_nbins_per_axis = np.ones_like(bins_per_axis, dtype=int)
    shifted_nbins_per_axis[:-1] = bins_per_axis[1:]
    # for each non-uniform point, this gives the linearized coordinate of the
    # appropriate bin
    linearized_bin_coords = np.dot(bin_coords, shifted_nbins_per_axis[:, None])
    aug_linearized_bin_coords = np.hstack(
        [linearized_bin_coords, np.arange(len(pts))[:, None]]
    )

    # group by puts into the same group those points which are in the same bin.
    # anyway points in different bins cannot be in the same future
    indexes_inside_bins = group_by(aug_linearized_bin_coords)

    if pts_per_future != -1:
        # indexes inside bins splitted according to pts_per_future
        subgroups_inside_bins = list(
            map(
                # here we apply the finer granularity (n. of pts per future)
                lambda arr: np.split(arr, pts_per_future),
                map(np.array, indexes_inside_bins),
            )
        )
    else:
        subgroups_inside_bins = [list(map(np.array, indexes_inside_bins))]

    subgroups_coords_fu = client.scatter(
        [
            (pts[subgroup], bin_coords[subgroup[0]], subgroup)
            for bin_content in subgroups_inside_bins
            for subgroup in bin_content
        ]
    )

    return subgroups_coords_fu


def generate_padded_bin(
    bin_coords,
    uniform_grid_cell_size,
    uniform_grid_cell_count,
    bins_size,
    max_distance,
):
    # number of uniform cells needed to cover max_distance
    max_distance_in_cells = (max_distance // uniform_grid_cell_size).astype(
        int
    )
    max_distance_in_cells[max_distance % uniform_grid_cell_size > 1.0e-12] += 1

    n_axes = len(bins_size)
    # generate the uniform grid
    axes = tuple(
        np.concatenate(
            (
                # left padding
                np.linspace(
                    bin_coords[i] * bins_size[i] * uniform_grid_cell_size[i]
                    - max_distance,
                    bin_coords[i] * bins_size[i] * uniform_grid_cell_size[i],
                    max_distance_in_cells[i],
                ),
                # body
                np.arange(
                    bin_coords[i] * bins_size[i],
                    (bin_coords[i] + 1) * bins_size[i],
                )
                * uniform_grid_cell_size[i],
                # right padding
                np.linspace(
                    # the trailing +1 is due to the fact that we start one cell
                    # after the end of the bin
                    ((bin_coords[i] + 1) * bins_size[i] + 1)
                    * uniform_grid_cell_size[i],
                    ((bin_coords[i] + 1) * bins_size[i])
                    * uniform_grid_cell_size[i]
                    + max_distance
                    + 1.0e-10,
                    max_distance_in_cells[i],
                ),
            )
        )
        for i in range(n_axes)
    )
    return np.reshape(np.meshgrid(*axes), (-1, n_axes))


def generate_reference_padded_bin(
    uniform_grid_cell_size, uniform_grid_cell_count, bins_size, max_distance
):
    n_axes = len(bins_size)
    reference_bin_coords = np.zeros(len(bins_size), dtype=int)
    return generate_padded_bin(
        reference_bin_coords,
        uniform_grid_cell_size,
        uniform_grid_cell_count,
        bins_size,
        max_distance,
    )


def compute_padded_bin_samples1_idxes(
    bin_coords, bins_size, uniform_grid_cell_count, max_distance_in_cells
):
    n_axes = len(bin_coords)
    # the +1 is for samples2
    axes = np.fromiter(range(n_axes + 1), dtype=int)
    return tuple(
        # we add a shallow dimension to all the dimension except for the
        # current one
        np.expand_dims(
            np.arange(
                max(
                    0, bin_coords[i] * bins_size[i] - max_distance_in_cells[i]
                ),
                min(
                    (bin_coords[i] + 1) * bins_size[i]
                    + max_distance_in_cells[i],
                    uniform_grid_cell_count[i],
                ),
            ),
            axis=tuple(np.delete(axes, i)),
        )
        for i in range(n_axes)
    )


def compute_mapped_distance_on_subgroup(
    subgroups_and_coords,
    uniform_grid_cell_count,
    uniform_grid_cell_size,
    reference_padded_bin,
    bins_size,
    max_distance,
    function,
    exact_max_distance,
):
    subgroup, bin_coords, samples2_idxes = subgroups_and_coords
    # we translate the reference

    max_distance_in_cells = (max_distance // uniform_grid_cell_size).astype(
        int
    )
    samples1 = (
        reference_padded_bin
        + (bin_coords * bins_size * uniform_grid_cell_size)[None]
    )
    samples1_idxes = compute_padded_bin_samples1_idxes(
        bin_coords,
        bins_size,
        uniform_grid_cell_count,
        max_distance_in_cells,
    )
    samples2_idxes = np.expand_dims(
        samples2_idxes, axis=tuple(range(len(bins_size)))
    )

    distances = np.linalg.norm(
        samples1[:, None, :] - subgroup[None, ...],
        axis=-1,
    )

    if exact_max_distance:
        nearby = distances < max_distance
        mapped_distance = np.zeros_like(distances, dtype=subgroup.dtype)
        mapped_distance[nearby] = function(distances[nearby])
    else:
        mapped_distance = function(distances)

    return mapped_distance, (*samples1_idxes, samples2_idxes)


def mapped_distance_matrix(
    # dimension of each cell, for each axis
    uniform_grid_cell_size,
    # number of cell, for each axis
    uniform_grid_cell_count,
    samples2,
    # dimension of the bins in terms of cells, for each axis
    bins_size,
    max_distance,
    func,
    client,
    should_vectorize=True,
    exact_max_distance=True,
    pts_per_future=5,
):
    # not using np.vectorize if the function is already vectorized allows us
    # to save some time
    if should_vectorize:
        func = np.vectorize(func)

    if len(bins_size) != samples2.shape[1]:
        raise ValueError(
            "Expected one bin-size per axis (received {})".format(
                len(bins_size)
            )
        )

    subgroups_coords_fu = fill_bins(
        uniform_grid_cell_size,
        uniform_grid_cell_count,
        samples2,
        bins_size,
        pts_per_future=pts_per_future,
        client=client,
    )

    # the bin whose bin_coord is (0,0,...,0), padded with max_distance
    reference_padded_bin = generate_reference_padded_bin(
        uniform_grid_cell_size,
        uniform_grid_cell_count,
        bins_size,
        max_distance,
    )
    print(reference_padded_bin.shape)

    mapped_distance = np.zeros(
        (*uniform_grid_cell_count, len(samples2)),
        dtype=samples2.dtype,
    )
    # all the writes to the global distance matrix occur in the main thread
    mapped_distances_fu = client.map(
        compute_mapped_distance_on_subgroup,
        subgroups_coords_fu,
        uniform_grid_cell_count=uniform_grid_cell_count,
        uniform_grid_cell_size=uniform_grid_cell_size,
        reference_padded_bin=reference_padded_bin,
        bins_size=bins_size,
        max_distance=max_distance,
        exact_max_distance=exact_max_distance,
        function=func,
    )

    for _, (submatrix, idxes_broadcastable_tuple) in as_completed(
        mapped_distances_fu, with_results=True
    ):
        mapped_distance[idxes_broadcastable_tuple] = submatrix

    return mapped_distance
