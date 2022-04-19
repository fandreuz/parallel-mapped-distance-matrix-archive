from pycsou.linop.sampling import MappedDistanceMatrix
from time import time

from big_data import samples1, samples2, func

start = time()
MappedDistanceMatrix(
    samples1=samples1,
    samples2=samples2,
    function=func,
    operator_type="sparse",
    max_distance=1,
    n_jobs=4
).mat.compute()
print(time() - start)
