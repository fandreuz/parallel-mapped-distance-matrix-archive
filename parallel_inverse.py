import numpy as np

from dask.distributed import as_completed


def group_by(a):
    a = a[a[:, 0].argsort()]
    return np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:])


def bin_content_tuple(bin_content, pts, bin_coords):
    bin_content = bin_content[0]
    return pts[bin_content], bin_coords[bin_content[0]], bin_content


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
                lambda arr: np.array_split(
                    arr, np.ceil(len(arr) / pts_per_future)
                ),
                indexes_inside_bins,
            )
        )
    else:
        subgroups_inside_bins = [list(map(np.array, indexes_inside_bins))]

    bin_coords_fu = client.scatter(bin_coords, broadcast=True)
    pts_fu = client.scatter(pts, broadcast=True)
    subgroups_coords_fu = client.map(
        bin_content_tuple,
        tuple(
            bin_content
            for bin_content in subgroups_inside_bins
            for subgroup in bin_content
        ),
        pts=pts_fu,
        bin_coords=bin_coords_fu,
    )

    return subgroups_coords_fu


def compute_padded_bin_samples1_idxes(
    bin_coords, bins_size, uniform_grid_cell_count, max_distance_in_cells, clip
):
    lower_bound = bin_coords * bins_size - max_distance_in_cells
    upper_bound = (bin_coords + 1) * bins_size + max_distance_in_cells + 1

    if clip:
        lower_bound = np.clip(lower_bound, 0, None)
        upper_bound = np.clip(upper_bound, None, uniform_grid_cell_count)

    sliceify = lambda a: slice(a[0], a[1])
    return np.apply_along_axis(
        sliceify, 0, np.stack((lower_bound, upper_bound))
    )


def generate_padded_bin(slices, uniform_grid_cell_size):
    return np.mgrid[slices] * uniform_grid_cell_size[:, None, None]


def compute_mapped_distance_on_subgroup(
    subgroups_and_coords,
    uniform_grid_cell_count,
    uniform_grid_cell_size,
    bins_size,
    max_distance,
    function,
    exact_max_distance,
):
    subgroup, bin_coords, samples2_idxes = subgroups_and_coords
    # we translate the reference

    max_distance_in_cells = np.ceil(
        max_distance / uniform_grid_cell_size
    ).astype(int)

    uniform_grid_cell_count = np.asarray(uniform_grid_cell_count)

    samples1_idxes = compute_padded_bin_samples1_idxes(
        bin_coords,
        bins_size,
        uniform_grid_cell_count,
        max_distance_in_cells,
        clip=True,
    )

    samples1 = generate_padded_bin(
        slices=samples1_idxes,
        uniform_grid_cell_size=uniform_grid_cell_size,
    )

    distances = np.linalg.norm(
        samples1[..., None] - subgroup.T[:, None, None],
        axis=0,
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

    # all the writes to the global distance matrix occur in the main thread
    mapped_distances_fu = client.map(
        compute_mapped_distance_on_subgroup,
        subgroups_coords_fu,
        uniform_grid_cell_count=uniform_grid_cell_count,
        uniform_grid_cell_size=uniform_grid_cell_size,
        bins_size=bins_size,
        max_distance=max_distance,
        exact_max_distance=exact_max_distance,
        function=func,
    )

    mapped_distance = np.zeros(
        (*uniform_grid_cell_count, len(samples2)),
        dtype=samples2.dtype,
    )
    for _, (submatrix, idxes_broadcastable_tuple) in as_completed(
        mapped_distances_fu, with_results=True
    ):
        mapped_distance[idxes_broadcastable_tuple] = submatrix

    return mapped_distance.reshape(-1, len(samples2))
