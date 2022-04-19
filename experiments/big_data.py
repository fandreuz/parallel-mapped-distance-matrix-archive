import numpy as np

t = np.linspace(0, 2, 1000)
rng = np.random.default_rng(seed=2)
x, y = np.meshgrid(t, t)
samples1 = np.stack((x.flatten(), y.flatten()), axis=-1)
samples2 = np.stack(
    (2 * rng.random(size=100), 2 * rng.random(size=100)), axis=-1
)

sigma = 1 / 12
def func(x):
    return np.exp(-(x ** 2) / (2 * sigma ** 2))
