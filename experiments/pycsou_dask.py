from pycsou.linop.sampling import MappedDistanceMatrix
from time import time
import sys

if int(sys.argv[1]) == 0:
    from data import samples1, samples2, func
elif int(sys.argv[1]) == 1:
    from big_data import samples1, samples2, func
elif int(sys.argv[1]) == 2:
    from biggest_data import samples1, samples2, func
elif int(sys.argv[1]) == 3:
    from biggest_data2 import samples1, samples2, func
elif int(sys.argv[1]) >= 4:
    from scattered import samples1, samples2, func

start = time()
MappedDistanceMatrix(
    samples1=samples1,
    samples2=samples2,
    function=func,
    operator_type="dask",
    max_distance=float(sys.argv[2]),
    dtype=samples1.dtype
).mat.compute()
print(time() - start)

