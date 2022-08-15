from time import time
import sys
import numpy as np
from importlib import import_module

RUNNER = 1
N_ITERATIONS = 2
DATA = 3
MAX_DISTANCE = 4
# the scale goes from 1 to 100
PROBLEM_DIM = 5

bins_size = np.array([10, 10])
data = import_module(sys.argv[DATA])
problem_dim = float(sys.argv[PROBLEM_DIM])
samples1 = data.samples1(problem_dim)
samples2 = data.samples2(problem_dim)
func = data.func
uniform_params = data.uniform_params(problem_dim)
max_distance = float(sys.argv[MAX_DISTANCE])
iterations = int(sys.argv[N_ITERATIONS])
runner = import_module(sys.argv[RUNNER])


def after_dot(s):
    return s[s.rindex(".") + 1 :]


def run():
    times = np.zeros(iterations, dtype=np.single)
    ok = np.full(iterations, True)
    for i in range(iterations):
        start = time()
        try:
            runner.run(
                samples1,
                samples2,
                func,
                max_distance,
                uniform_params,
                bins_size,
            )
        except:
            print("Error: " + str(sys.argv) + ", ignoring...")
            ok[i] = False
            continue
        times[i] = time() - start

    if np.count_nonzero(ok) == 0:
        run()
    else:
        mean = np.mean(times[ok])
        var = np.var(times[ok], ddof=1)
        m = np.min(times[ok])
        M = np.max(times[ok])

        np.save(
            "results/{}_{}_{}_{}.npy".format(
                after_dot(sys.argv[RUNNER]),
                after_dot(sys.argv[DATA]),
                sys.argv[MAX_DISTANCE],
                problem_dim,
            ),
            np.array([mean, var, m, M]),
        )


run()
