import numpy as np
import matplotlib.pyplot as plt
from pycsou.linop.sampling import MappedDistanceMatrix

import sys
import os
sys.path.append(os.getcwd())
from serial import mapped_distance_matrix

from time import time

if __name__ == "__main__":
    t = np.linspace(0, 2, 500)
    rng = np.random.default_rng(seed=2)
    x, y = np.meshgrid(t, t)
    samples1 = np.stack((x.flatten(), y.flatten()), axis=-1)
    samples2 = np.stack(
        (2 * rng.random(size=50), 2 * rng.random(size=50)), axis=-1
    )
    alpha = np.ones(samples2.shape[0])
    sigma = 1 / 12
    func = lambda x: np.exp(-(x ** 2) / (2 * sigma ** 2))

    start = time()
    MDMOp = MappedDistanceMatrix(
        samples1=samples1,
        samples2=samples2,
        function=func,
        operator_type="dask",
    )
    print('pycsou: {} seconds'.format(time() - start))

    start = time()
    m = mapped_distance_matrix(samples1, samples2, 1, func, bins_per_axis=[5,5], should_vectorize=False)
    print('new: {} seconds'.format(time() - start))

    plt.contourf(x, y, m.dot(alpha).reshape(t.size, t.size), 50)
    plt.title("Sum of 4 (radial) Gaussians")
    plt.colorbar()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()
