import numpy as np
import cupy as cp
import time
import os
import math

print("Matrix: {}".format("1000x1000x1000"))

### Numpy and CPU
s = time.time()
x_cpu = np.ones((1000,1000,1000))
e = time.time()
print("CPU: {}".format(e - s))

### CuPy and GPU
s = time.time()
x_gpu = cp.ones((1000,1000,1000))
cp.cuda.Stream.null.synchronize()
e = time.time()
print("GPU: {}".format(e - s))


print("\nMatrix: {}".format("x5"))

### Numpy and CPU
s = time.time()
x_cpu *= 5
e = time.time()
print("CPU: {}".format(e - s))

### CuPy and GPU
s = time.time()
x_gpu *= 5
cp.cuda.Stream.null.synchronize()
e = time.time()
print("GPU {}".format(e - s))

print("\nMatrix: {}".format("x5 ^2"))

### Numpy and CPU
s = time.time()
x_cpu *= 5
x_cpu *= x_cpu
x_cpu += x_cpu
e = time.time()
print("CPU: {}".format(e - s))


### CuPy and GPU
s = time.time()
x_gpu *= 5
x_gpu *= x_gpu
x_gpu += x_gpu
cp.cuda.Stream.null.synchronize()
e = time.time()
print("GPU {}".format(e - s))

repeats = 6
iterations = 120


### CuPy and GPU
def run_gpu(x_gpu, iterations):
    s = time.time()
    for x in range(1,iterations):
        x_gpu *= 5
        x_gpu *= x_gpu
        x_gpu += x_gpu
        if x % math.ceil((iterations/2)+0.1) == 0:
            os.system('nvidia-smi')
            print("Still running...")
    cp.cuda.Stream.null.synchronize()
    e = time.time()
    print("GPU: {}".format(e - s))


print("Repeating for iterations: [{}]".format(str(repeats)))
print("\nMatrix: {} * {}".format("x5 ^2",str(iterations)))

for y in range(1,6):
    run_gpu(x_gpu, iterations)
    print("Completed cycle: [{}]".format(y))