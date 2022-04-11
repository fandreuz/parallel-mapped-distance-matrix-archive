from time import time
from pycsou.linop.sampling import MappedDistanceMatrix
import numpy as np
from tqdm import tqdm

import sys, os

sys.path.append(os.getcwd())
from parallel_vstack import mapped_distance_matrix as vmapped_distance_matrix
from parallel import mapped_distance_matrix


sigma = 1 / 12


def f(x):
    return np.exp(-(x ** 2) / (2 * sigma ** 2))


def pycsou(samples1, samples2, func=f):
    MappedDistanceMatrix(
        samples1=samples1,
        samples2=samples2,
        function=func,
        operator_type="dask",
    ).mat.compute()


# ----- BINS


def bins_md1_bpa55(samples1, samples2, func=f):
    mapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
    ).compute()


def bins_md04_bpa55(samples1, samples2, func=f):
    mapped_distance_matrix(
        samples1,
        samples2,
        0.4,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
    ).compute()


def bins_md1_bpa1010(samples1, samples2, func=f):
    mapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
    ).compute()


def bins_md04_bpa1010(samples1, samples2, func=f):
    mapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
    ).compute()


def bins_md1_bpa55_emdf(samples1, samples2, func=f):
    mapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


def bins_md04_bpa55_emdf(samples1, samples2, func=f):
    mapped_distance_matrix(
        samples1,
        samples2,
        0.4,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


def bins_md1_bpa1010_emdf(samples1, samples2, func=f):
    mapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


def bins_md04_bpa1010_emdf(samples1, samples2, func=f):
    mapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


# -------- VSTACK


def vbins_md1_bpa55(samples1, samples2, func=f):
    vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
    ).compute()


def vbins_md04_bpa55(samples1, samples2, func=f):
    vmapped_distance_matrix(
        samples1,
        samples2,
        0.4,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
    ).compute()


def vbins_md1_bpa1010(samples1, samples2, func=f):
    vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
    ).compute()


def vbins_md04_bpa1010(samples1, samples2, func=f):
    vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
    ).compute()


def vbins_md1_bpa55_emdf(samples1, samples2, func=f):
    vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


def vbins_md04_bpa55_emdf(samples1, samples2, func=f):
    vmapped_distance_matrix(
        samples1,
        samples2,
        0.4,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


def vbins_md1_bpa1010_emdf(samples1, samples2, func=f):
    vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


def vbins_md04_bpa1010_emdf(samples1, samples2, func=f):
    vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


# ------ bins_array_chunks


def vbins_md1_bpa55_baca(samples1, samples2, func=f):
    vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
        bins_array_chunks="auto",
    ).compute()


def vbins_md04_bpa55_baca(samples1, samples2, func=f):
    vmapped_distance_matrix(
        samples1,
        samples2,
        0.4,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
        bins_array_chunks="auto",
    ).compute()


def vbins_md1_bpa1010_baca(samples1, samples2, func=f):
    vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
        bins_array_chunks="auto",
    ).compute()


def vbins_md04_bpa1010_baca(samples1, samples2, func=f):
    vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
        bins_array_chunks="auto",
    ).compute()


def vbins_md1_bpa55_emdf_baca(samples1, samples2, func=f):
    vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
        exact_max_distance=False,
        bins_array_chunks="auto",
    ).compute()


def vbins_md04_bpa55_emdf_baca(samples1, samples2, func=f):
    vmapped_distance_matrix(
        samples1,
        samples2,
        0.4,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
        exact_max_distance=False,
        bins_array_chunks="auto",
    ).compute()


def vbins_md1_bpa1010_emdf_baca(samples1, samples2, func=f):
    vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
        exact_max_distance=False,
        bins_array_chunks="auto",
    ).compute()


def vbins_md04_bpa1010_emdf_baca(samples1, samples2, func=f):
    vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
        exact_max_distance=False,
        bins_array_chunks="auto",
    ).compute()


benchmarks = [
    pycsou,
    vbins_md1_bpa55,
    vbins_md04_bpa55,
    vbins_md1_bpa1010,
    vbins_md04_bpa1010,
    vbins_md1_bpa55_emdf,
    vbins_md04_bpa55_emdf,
    vbins_md1_bpa1010_emdf,
    vbins_md04_bpa1010_emdf,
    vbins_md1_bpa55_baca,
    vbins_md04_bpa55_baca,
    vbins_md1_bpa1010_baca,
    vbins_md04_bpa1010_baca,
    vbins_md1_bpa55_emdf_baca,
    vbins_md04_bpa55_emdf_baca,
    vbins_md1_bpa1010_emdf_baca,
    vbins_md04_bpa1010_emdf_baca,
]


def do_benchmark(samples1, samples2):
    time_measured = []
    labels = []
    for bc in tqdm(benchmarks):
        labels.append(bc.__name__)
        start = time()
        bc(samples1, samples2)
        time_measured.append(time() - start)

    print("Results:")
    for label, t in zip(labels, time_measured):
        print("- {} : {} seconds".format(label, t))
