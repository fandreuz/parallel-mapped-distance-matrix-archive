from pycsou.linop.sampling import MappedDistanceMatrix


def run(samples1, samples2, func, max_distance, uniform_params, bins_size):
    return MappedDistanceMatrix(
        samples1=samples1,
        samples2=samples2,
        function=func,
        operator_type="dask",
        max_distance=max_distance,
        dtype=samples2.dtype,
    ).mat.compute()
