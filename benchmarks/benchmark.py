from time import time
from pycsou.linop.sampling import MappedDistanceMatrix
import numpy as np
from tqdm import tqdm

import sys, os

sys.path.append(os.getcwd())
from parallel_map_blocks import mapped_distance_matrix as mbmapped_distance_matrix
from parallel_futures import mapped_distance_matrix as fmapped_distance_matrix


sigma = 1 / 12


def f(x):
    return np.exp(-(x ** 2) / (2 * sigma ** 2))


def pycsou(samples1, samples2, func=f, **kwargs):
    return MappedDistanceMatrix(
        samples1=samples1,
        samples2=samples2,
        function=func,
        operator_type="dask",
    ).mat.compute()


# ----- BINS


def mbbins_md1_bpa55(samples1, samples2, func=f):
    return mbmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
    ).compute()


def mbbins_md04_bpa55(samples1, samples2, func=f):
    return mbmapped_distance_matrix(
        samples1,
        samples2,
        0.4,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
    ).compute()


def mbbins_md1_bpa1010(samples1, samples2, func=f):
    return mbmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
    ).compute()


def mbbins_md04_bpa1010(samples1, samples2, func=f):
    return mbmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
    ).compute()


def mbbins_md1_bpa55_emdf(samples1, samples2, func=f):
    return mbmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


def mbbins_md04_bpa55_emdf(samples1, samples2, func=f):
    return mbmapped_distance_matrix(
        samples1,
        samples2,
        0.4,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


def mbbins_md1_bpa1010_emdf(samples1, samples2, func=f):
    return mbmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


def mbbins_md04_bpa1010_emdf(samples1, samples2, func=f):
    return mbmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


# ----- binsperchunk

def mbbins_md1_bpa44_bpc4(samples1, samples2, func=f):
    return mbmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[4,4],
        should_vectorize=False,
        bins_per_chunk=4
    ).compute()


def mbbins_md04_bpa44_bpc4(samples1, samples2, func=f):
    return mbmapped_distance_matrix(
        samples1,
        samples2,
        0.4,
        func,
        bins_per_axis=[4,4],
        should_vectorize=False,
        bins_per_chunk=4
    ).compute()


def mbbins_md1_bpa66_bpc9(samples1, samples2, func=f):
    return mbmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[6, 6],
        should_vectorize=False,
        bins_per_chunk=9
    ).compute()


def mbbins_md04_bpa66_bpc9(samples1, samples2, func=f):
    return mbmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[6, 6],
        should_vectorize=False,
        bins_per_chunk=9
    ).compute()


def mbbins_md1_bpa44_emdf_bpc4(samples1, samples2, func=f, **kwargs):
    return mbmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[4, 4],
        should_vectorize=False,
        exact_max_distance=False,
        bins_per_chunk=4
    ).compute()


def mbbins_md04_bpa44_emdf_bpc4(samples1, samples2, func=f, **kwargs):
    return mbmapped_distance_matrix(
        samples1,
        samples2,
        0.4,
        func,
        bins_per_axis=[4, 4],
        should_vectorize=False,
        exact_max_distance=False,
        bins_per_chunk=4
    ).compute()


def mbbins_md1_bpa66_emdf_bpc9(samples1, samples2, func=f, **kwargs):
    return mbmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[6, 6],
        should_vectorize=False,
        exact_max_distance=False,
        bins_per_chunk=9
    ).compute()


def mbbins_md04_bpa66_emdf_bpc9(samples1, samples2, func=f, **kwargs):
    return mbmapped_distance_matrix(
        samples1,
        samples2,
        0.4,
        func,
        bins_per_axis=[6, 6],
        should_vectorize=False,
        exact_max_distance=False,
        bins_per_chunk=9
    ).compute()


# ----------


def future_1010_5(samples1, samples2,func=f,client=None):
    return fmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        client=client,
        bins_per_axis=[10,10],
        should_vectorize=False,
        exact_max_distance=False,
        bins_per_future=5
    )

def future_1010_10(samples1, samples2,func=f,client=None):
    return fmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        client=client,
        bins_per_axis=[50,50],
        should_vectorize=False,
        exact_max_distance=False,
        bins_per_future=10
    )

def future_55_5(samples1, samples2,func=f,client=None):
    return fmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        client=client,
        bins_per_axis=[5,5],
        should_vectorize=False,
        exact_max_distance=False,
        bins_per_future=5
    )

def future_1010_5_s(samples1, samples2,func=f,client=None):
    return fmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        client=client,
        bins_per_axis=[10,10],
        should_vectorize=False,
        exact_max_distance=False,
        bins_per_future=5,
        sort_bins=True
    )

def future_1010_10_s(samples1, samples2,func=f,client=None):
    return fmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        client=client,
        bins_per_axis=[50,50],
        should_vectorize=False,
        exact_max_distance=False,
        bins_per_future=10,
        sort_bins=True
    )

def future_55_5_s(samples1, samples2,func=f,client=None):
    return fmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        client=client,
        bins_per_axis=[5,5],
        should_vectorize=False,
        exact_max_distance=False,
        bins_per_future=5,
        sort_bins=True
    )

benchmarks = [
    pycsou,
    future_1010_5,
    future_1010_10,
    future_55_5,
    future_1010_5_s,
    future_1010_10_s,
    future_55_5_s,
    mbbins_md1_bpa44_emdf_bpc4,
    mbbins_md04_bpa44_emdf_bpc4,
    mbbins_md1_bpa66_emdf_bpc9,
    mbbins_md04_bpa66_emdf_bpc9,
]


def do_benchmark(samples1, samples2, client=None):
    time_measured = []
    labels = []
    equal = []

    sample = None
    for bc in tqdm(benchmarks):
        labels.append(bc.__name__)
        start = time()
        mat = bc(samples1, samples2, client=client)
        time_measured.append(time() - start)

        if sample is None:
            sample = mat
            equal.append("SA")
        else:
            equal.append("EQ" if np.allclose(sample, mat, atol=1.e-16) else "NEQ {}".format(np.max(np.abs(sample - mat))))

    print("Results:")
    for label, t, eq in zip(labels, time_measured, equal):
        print("- {} : {} seconds ({})".format(label, t, eq))
