import numpy as np
from benchmark import do_benchmark

if __name__ == "__main__":
    t = np.linspace(0, 2, 500)
    rng = np.random.default_rng(seed=2)
    x, y = np.meshgrid(t, t)
    samples1 = np.stack((x.flatten(), y.flatten()), axis=-1)
    samples2 = np.stack(
        (2 * rng.random(size=50), 2 * rng.random(size=50)), axis=-1
    )

    do_benchmark(samples1, samples2)
