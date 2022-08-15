import sys
import pathlib

sys.path.append(str(pathlib.Path().resolve().parent) + "/src")

from numba_loops import mapped_distance_matrix

from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(8)


def run(samples1, samples2, func, max_distance, uniform_params, bins_size):
    return mapped_distance_matrix(
        uniform_grid_cell_step=uniform_params["grid_step"],
        uniform_grid_size=uniform_params["grid_size"],
        bins_size=bins_size,
        non_uniform_points=samples2,
        max_distance=max_distance,
        func=func,
        executor=executor,
        exact_max_distance=True,
    )
