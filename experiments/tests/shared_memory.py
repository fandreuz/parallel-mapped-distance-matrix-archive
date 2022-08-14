from dask.distributed import Client
import numpy as np

def fill(idx, matrix):
    matrix[idx] = idx
    print(idx, matrix[idx])

if __name__ == '__main__':
    client = Client()

    N = 20
    m = np.zeros((N, 10))
    client.gather(client.map(fill, range(N), matrix=m))

    print('-----')
    print('11th row is', m[10])
    print(np.count_nonzero(m))
