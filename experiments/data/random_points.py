import numba as nb
import numpy as np


def samples1(N):
    dc = uniform_params(N)
    return (
        np.swapaxes(
            np.mgrid[: dc["grid_size"][0], : dc["grid_size"][1]].T, 0, 1
        )
        * dc["grid_step"]
    ).reshape((-1, 2))


def samples2(N):
    return np.random.rand(int(10000 * N), 2)


def uniform_params(N):
    dc = {}
    dc["grid_step"] = np.array([0.01, 0.01])
    dc["grid_size"] = np.array([100, 100])
    return dc


@nb.njit
def func(x):
    sigma = 1 / 12
    return np.exp(-(x ** 2) / (2 * sigma ** 2))
