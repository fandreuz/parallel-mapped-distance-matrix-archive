from string import Template
from itertools import product

t = Template("""
from pathlib import Path
import sys
sys.path.append(Path(__file__).parent.parent)

from parallel_map_blocks import mapped_distance_matrix
from time import time

if int(data[sys.argv[2]]) == 0:
    from data import samples1, samples2, func
elif int(data[sys.argv[2]]) == 1:
    from big_data import samples1, samples2, func
elif int(data[sys.argv[2]]) == 2:
    from biggest_data import samples1, samples2, func

start = time()
mapped_distance_matrix(
    samples1=samples1,
    samples2=samples2,
    max_distance=float(sys.argv[1]),
    function=func,
    bins_per_axis=[$bpa, $bpa],
    should_vectorize=False,
    exact_max_distance=False,
    bins_per_chunk=$bpc,
).compute()
print(time() - start)
""")

bpcs = [1, 2, 5, 10, 15, 20]
bpas = [1, 2, 5, 10, 15, 20, 50, 100, 250]

for bpa, bpc in product(bpas, bpcs):
    if bpc < bpa * bpa:
        f = open("pmb_{}_{}.py".format(bpa, bpc), "w")
        f.write(t.substitute({'bpa' : bpa, 'bpc': bpc}))
        f.close()
