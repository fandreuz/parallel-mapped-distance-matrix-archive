import numpy as np


def extract_slice(arr, lower_bounds, upper_bounds):
    r"""
    Extract a multi-dimensional slice from `arr`. No sanity check of parameters
    is performed, everything is done using NumPy indexing.

    Parameters
    ----------
    arr: np.ndarray
        The NumPy array which contains slices that we want to extract.
    lower_bounds: iterable
        An iterable which contains the lower bounds of the slices (included)
        for each dimension (`len(lower_bounds) == arr.ndims`).
    upper_bounds: iterable
        An iterable which contains the upper bounds of the slices (excluded)
        for each dimension (`len(lower_bounds) == arr.ndims`).

    Returns
    -------
    `np.ndarray`
    Slices extracted from `arr`.
    """

    # 1D
    # return arr[lower_bounds[0] : upper_bounds[0]]

    # 2D
    return arr[
        lower_bounds[0] : upper_bounds[0],
        lower_bounds[1] : upper_bounds[1],
    ]

    # 3D
    # return arr[
    #    lower_bounds[0] : upper_bounds[0],
    #    lower_bounds[1] : upper_bounds[1],
    #    lower_bounds[2] : upper_bounds[2],
    # ]

    # ND
    # return arr[
    #    tuple(slice(l, u) for l, u in zip(lower_bounds, upper_bounds))
    # ]


def add_to_slice(arr, val, lower_bounds, upper_bounds):
    r"""
    Add the given NumPy array `val` to a slice of `arr` defined by
    `lower_bounds` and `upper_bounds`. `arr` is modified in place.

    Parameters
    ----------
    arr: np.ndarray
        The NumPy array which contains slices that we want to sum to.
    val: np.ndarray
        The NumPy array to be summed to the slice.
    lower_bounds: iterable
        An iterable which contains the lower bounds of the slices (included)
        for each dimension (`len(lower_bounds) == arr.ndims`).
    upper_bounds: iterable
        An iterable which contains the upper bounds of the slices (excluded)
        for each dimension (`len(lower_bounds) == arr.ndims`).
    """

    # 1D
    # arr[lower_bounds[0] : upper_bounds[0]] += val

    # 2D
    arr[
        lower_bounds[0] : upper_bounds[0],
        lower_bounds[1] : upper_bounds[1],
    ] += val

    # 3D
    # arr[
    #    lower_bounds[0] : upper_bounds[0],
    #    lower_bounds[1] : upper_bounds[1],
    #    lower_bounds[2] : upper_bounds[2],
    # ] += val

    # ND
    # arr[
    #    tuple(slice(l, u) for l, u in zip(lower_bounds, upper_bounds))
    # ] += val


def periodic_inner_sum(arr, core_lower_bound, core_upper_bound):
    r"""
    Given a NumPy array `arr` which contains some external slices which should
    be "flipped" and summed periodically, perform the flip operation and return
    the core of `arr`.

    For instance, let's say that the first and last two rows, and the first
    and last three columns in `arr` are periodic parts of `arr`. This means
    that, for instance, `arr[[-1,-2], [-1,-2,-3]]` should be summed to
    `arr[[2,3], [3,4,5]]` in this order.

    The core is the internal, non periodic part. The periodic part is summed
    against the core.

    Parameters
    ----------
    arr: np.ndarray
        A NumPy array which contains periodic slices.
    core_lower_bound: iterable
        Lower bound (included) of the core of `arr`, for each dimension of
        `arr`.
    upper_bounds: iterable
        Upper bound (not included) of the core of `arr`, for each dimension of
        `arr`.
    """

    right_size = np.array(arr.shape) - core_upper_bound

    # 1D
    # arr[core_lower_bound[0] : 2 * core_lower_bound[0]] += arr[
    #     : core_lower_bound[0]
    # ]
    # arr[core_upper_bound[0] - right_size[0] : core_upper_bound[0]] += arr[
    #     core_upper_bound[0] :
    # ]
    # return arr[core_lower_bound[0] : core_upper_bound[0]]

    # 2D
    arr[
        core_lower_bound[0] : 2 * core_lower_bound[0],
        core_lower_bound[1] : 2 * core_lower_bound[1],
    ] += arr[: core_lower_bound[0], : core_lower_bound[1]]
    arr[
        core_upper_bound[0] - right_size[0] : core_upper_bound[0],
        core_upper_bound[1] - right_size[1] : core_upper_bound[1],
    ] += arr[core_upper_bound[0] :, core_upper_bound[1] :]
    return arr[
        core_lower_bound[0] : core_upper_bound[0],
        core_lower_bound[1] : core_upper_bound[1],
    ]

    # 3D
    # arr[
    #     core_lower_bound[0] : 2 * core_lower_bound[0],
    #     core_lower_bound[1] : 2 * core_lower_bound[1],
    #     core_lower_bound[2] : 2 * core_lower_bound[2],
    # ] += arr[
    #     : core_lower_bound[0], : core_lower_bound[1], : core_lower_bound[2]
    # ]
    # arr[
    #     core_upper_bound[0] - right_size[0] : core_upper_bound[0],
    #     core_upper_bound[1] - right_size[1] : core_upper_bound[1],
    #     core_upper_bound[2] - right_size[2] : core_upper_bound[2],
    # ] += arr[
    #     core_upper_bound[0] :, core_upper_bound[1] :, core_upper_bound[2] :
    # ]
    # return arr[
    #     core_lower_bound[0] : core_upper_bound[0],
    #     core_lower_bound[1] : core_upper_bound[1],
    #     core_lower_bound[2] : core_upper_bound[2],
    # ]

    # ND
    # arr[
    #     tuple(slice(bottom, 2 * bottom) for bottom in core_lower_bound)
    # ] += arr[tuple(slice(0, bottom) for bottom in core_lower_bound)]
    # arr[
    #     tuple(
    #         slice(top - rs, top)
    #         for top, rs in zip(core_upper_bound, right_size)
    #     )
    # ] += arr[tuple(slice(top, None) for top in core_upper_bound)]
    # return arr[
    #     tuple(
    #         slice(bottom, top)
    #         for bottom, top in zip(core_lower_bound, core_upper_bound)
    #     )
    # ]
