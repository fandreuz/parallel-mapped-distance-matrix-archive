from time import time
from pycsou.linop.sampling import MappedDistanceMatrix
import numpy as np
from tqdm import tqdm

import sys, os

sys.path.append(os.getcwd())
from parallel_vstack import mapped_distance_matrix as vmapped_distance_matrix
from parallel import mapped_distance_matrix
from parallel_map_blocks import mapped_distance_matrix as mbmapped_distance_matrix


sigma = 1 / 12


def f(x):
    return np.exp(-(x ** 2) / (2 * sigma ** 2))


def pycsou(samples1, samples2, func=f):
    return MappedDistanceMatrix(
        samples1=samples1,
        samples2=samples2,
        function=func,
        operator_type="dask",
    ).mat.compute()


# ----- BINS


def bins_md1_bpa55(samples1, samples2, func=f):
    return mapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
    ).compute()


def bins_md04_bpa55(samples1, samples2, func=f):
    return mapped_distance_matrix(
        samples1,
        samples2,
        0.4,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
    ).compute()


def bins_md1_bpa1010(samples1, samples2, func=f):
    return mapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
    ).compute()


def bins_md04_bpa1010(samples1, samples2, func=f):
    return mapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
    ).compute()


def bins_md1_bpa55_emdf(samples1, samples2, func=f):
    return mapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


def bins_md04_bpa55_emdf(samples1, samples2, func=f):
    return mapped_distance_matrix(
        samples1,
        samples2,
        0.4,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


def bins_md1_bpa1010_emdf(samples1, samples2, func=f):
    return mapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


def bins_md04_bpa1010_emdf(samples1, samples2, func=f):
    return mapped_distance_matrix(
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
    return vmapped_distance_matrix(
        samples1,
        samples2,
        0.4,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
    ).compute()


def vbins_md1_bpa1010(samples1, samples2, func=f):
    return vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
    ).compute()


def vbins_md04_bpa1010(samples1, samples2, func=f):
    return vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
    ).compute()


def vbins_md1_bpa55_emdf(samples1, samples2, func=f):
    return vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


def vbins_md04_bpa55_emdf(samples1, samples2, func=f):
    return vmapped_distance_matrix(
        samples1,
        samples2,
        0.4,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


def vbins_md1_bpa1010_emdf(samples1, samples2, func=f):
    return vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
        exact_max_distance=False,
    ).compute()


def vbins_md04_bpa1010_emdf(samples1, samples2, func=f):
    return vmapped_distance_matrix(
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
    return vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
        bins_array_chunks="auto",
    ).compute()


def vbins_md04_bpa55_baca(samples1, samples2, func=f):
    return vmapped_distance_matrix(
        samples1,
        samples2,
        0.4,
        func,
        bins_per_axis=[5, 5],
        should_vectorize=False,
        bins_array_chunks="auto",
    ).compute()


def vbins_md1_bpa1010_baca(samples1, samples2, func=f):
    return vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
        bins_array_chunks="auto",
    ).compute()


def vbins_md04_bpa1010_baca(samples1, samples2, func=f):
    return vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
        bins_array_chunks="auto",
    ).compute()


def vbins_md1_bpa55_emdf_baca(samples1, samples2, func=f):
    return vmapped_distance_matrix(
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
    return vmapped_distance_matrix(
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
    return vmapped_distance_matrix(
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
    return vmapped_distance_matrix(
        samples1,
        samples2,
        1,
        func,
        bins_per_axis=[10, 10],
        should_vectorize=False,
        exact_max_distance=False,
        bins_array_chunks="auto",
    ).compute()

# -------------------

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


def mbbins_md1_bpa44_emdf_bpc4(samples1, samples2, func=f):
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


def mbbins_md04_bpa44_emdf_bpc4(samples1, samples2, func=f):
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


def mbbins_md1_bpa66_emdf_bpc9(samples1, samples2, func=f):
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


def mbbins_md04_bpa66_emdf_bpc9(samples1, samples2, func=f):
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


benchmarks = [
    pycsou,
    mbbins_md1_bpa55,
    mbbins_md04_bpa55,
    mbbins_md1_bpa1010,
    mbbins_md04_bpa1010,
    mbbins_md1_bpa55_emdf,
    mbbins_md04_bpa55_emdf,
    mbbins_md1_bpa1010_emdf,
    mbbins_md04_bpa1010_emdf,
    mbbins_md1_bpa44_bpc4,
    mbbins_md04_bpa44_bpc4,
    mbbins_md1_bpa66_bpc9,
    mbbins_md04_bpa66_bpc9,
    mbbins_md1_bpa44_emdf_bpc4,
    mbbins_md04_bpa44_emdf_bpc4,
    mbbins_md1_bpa66_emdf_bpc9,
    mbbins_md04_bpa66_emdf_bpc9,
]


def do_benchmark(samples1, samples2):
    time_measured = []
    labels = []
    equal = []

    sample = None
    for bc in tqdm(benchmarks):
        labels.append(bc.__name__)
        start = time()
        mat = bc(samples1, samples2)
        time_measured.append(time() - start)

        if sample is None:
            sample = mat
            equal.append("SA")
        else:
            equal.append("EQ" if np.allclose(sample, mat, atol=1.e-16) else "NEQ {}".format(np.max(np.abs(sample - mat))))

    print("Results:")
    for label, t, eq in zip(labels, time_measured, equal):
        print("- {} : {} seconds ({})".format(label, t, eq))
