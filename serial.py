import numpy as np


# top-left and bottom-right
def bounds(bin_pts):
    return np.min(bin_pts, axis=0)[None, :], np.max(bin_pts, axis=0)[None, :]


# return a tensor of shape N x 2 x D where N is the number of bins, 2 is the
# number of bounds (top-left and bottom-right) and D is the dimensionality of
# the space
def compute_bins_bounds(bins, ndims):
    nbins = len(bins)
    numpy_bins = list(map(np.array, bins))

    bin_bounds = np.zeros((nbins, 2, ndims), dtype=float)
    for bin_idx in range(nbins):
        b = numpy_bins[bin_idx]
        # we don't do anything if the bin is empty
        if len(b) > 0:
            top_left, bottom_right = bounds(b)
            bin_bounds[bin_idx, 0] = top_left
            bin_bounds[bin_idx, 1] = bottom_right
    return numpy_bins, bin_bounds


def group_by(a):
    a = a[a[:, 0].argsort()]
    return np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:])


# build a list of lists, where each list contains the points in pts contained
# inside the bin corresponding to a certain (linearized) coordinate. For more on
# linearized bin coordinates see bin_coords and linearized_bin_coords.
# pts is the matrix of points.
# h is the granularity of the uniform grid we intend to build.
# bin_dims is the number of uniform points to be included in each (non-padded)
#   bin, in each direction.
# region_dimension is the dimension of the region used to enclose the points.
# it is preferable that bin_dims * h divides region_dimension exactly in each
# direction.
def fill_bins(pts, bins_per_axis, region_dimension):
    nbins = np.prod(bins_per_axis)
    h = np.divide(region_dimension, bins_per_axis)

    indexes_inside_bins = [[] for _ in range(nbins)]
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

    bins = []
    for indexes_in_bin in indexes_inside_bins:
        bins.append(pts[indexes_in_bin])

    return bins, indexes_inside_bins


def compute_padded_bin_bounds(boundaries, distance):
    return boundaries - (distance * np.array([1, -1]))[None, :, None]


def match_points_and_bins(bins_bounds, points):
    broadcastable_points = points[:, None]
    return np.logical_and(
        np.all(bins_bounds[:, 0] < broadcastable_points, axis=2),
        np.all(broadcastable_points < bins_bounds[:, 1], axis=2),
    )


def compute_distance(pts1, pts2):
    return np.linalg.norm(pts1[:, None, :] - pts2[None, ...], axis=-1)


def compute_mapped_distance_matrix(
    bins,
    indexes_inside_bins,
    pts1,
    pts2,
    inclusion_matrix,
    max_distance,
    func,
    exact_max_distance=True,
):
    # we filter away empty bins
    matrix = np.zeros((len(pts1), len(pts2)), dtype=float)

    for bin_idx in range(inclusion_matrix.shape[1]):
        bin_pts1 = bins[bin_idx]
        if len(bin_pts1) == 0:
            continue

        pts2_in_bin = inclusion_matrix[:, bin_idx]
        padded_bin_pts2 = pts2[pts2_in_bin]
        if len(padded_bin_pts2) == 0:
            continue

        bin_pts2_indexing_to_full = np.arange(len(pts2))[pts2_in_bin]

        distances = compute_distance(bin_pts1, padded_bin_pts2)

        # indexes is the list of indexes in pts1 that belong to this bin
        indexes = np.asarray(indexes_inside_bins[bin_idx])

        if exact_max_distance:
            nearby = distances < max_distance
        else:
            nearby = np.full(distances.shape, True)

        mapped_distance = func(distances)
        mapped_distance[np.logical_not(nearby)] = 0
        matrix[
            indexes[None, :], bin_pts2_indexing_to_full[:, None]
        ] = np.squeeze(mapped_distance).T

    return matrix


def mapped_distance_matrix(
    pts1,
    pts2,
    max_distance,
    func,
    bins_per_axis=None,
    chunks="auto",
    should_vectorize=True,
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

    ndims = pts1.shape[1]

    bins, indexes_inside_bins = fill_bins(
        pts1, bins_per_axis, region_dimension
    )
    bins, bins_bounds = compute_bins_bounds(bins, ndims)
    padded_bin_bounds = compute_padded_bin_bounds(bins_bounds, max_distance)
    assert padded_bin_bounds.shape == bins_bounds.shape

    inclusion_matrix = match_points_and_bins(padded_bin_bounds, pts2)
    mapped_distance = compute_mapped_distance_matrix(
        bins,
        indexes_inside_bins,
        pts1,
        pts2,
        inclusion_matrix,
        max_distance,
        func,
    )
    assert mapped_distance.shape == (len(pts1), len(pts2))

    return mapped_distance
