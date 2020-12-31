from time import perf_counter
import numpy as np

x = np.random.rand(256, 256)
y = np.random.rand(256, 256)

start = perf_counter()
z = np.dot(x, y)
end = perf_counter()
print(end-start)