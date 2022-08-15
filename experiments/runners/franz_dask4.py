import sys
import pathlib

sys.path.append(str(pathlib.Path().resolve().parent) + "/src")

from dask_client import mapped_distance_matrix

from dask.distributed import Client

client = Client(processes=False, n_workers=1, threads_per_worker=4)
client.restart()


def run(samples1, samples2, func, max_distance, uniform_params, bins_size):
    return mapped_distance_matrix(
        uniform_grid_cell_step=uniform_params["grid_step"],
        uniform_grid_size=uniform_params["grid_size"],
        bins_size=bins_size,
        non_uniform_points=samples2,
        max_distance=max_distance,
        func=func,
        client=client,
        pts_per_future=-1,
        exact_max_distance=True,
    )
