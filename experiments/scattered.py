import numpy as np
import sys

samples1 = np.random.rand(1500*1500, 2)
samples2 = np.random.rand(200, 2)

clusters = int(sys.argv[1]) - 2

part1 = len(samples1) // clusters
part2 = len(samples2) // clusters

for i in range(1,clusters):
	addition = np.zeros_like(samples1[0])
	addition[0] += i*100
	samples2[i*part2:(i+1)*part2] += addition
	samples1[i*part1:(i+1)*part1] += addition

sigma = 1 / 12
def func(x):
    return np.exp(-(x ** 2) / (2 * sigma ** 2))
