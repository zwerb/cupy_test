import numpy as np
import cupy as cp
import time
import os
import math


x_gpu = cp.ones((1000,1000,1000))
iterations = 5
multiplier = 200

print("Matrix: {}".format("cp.ones((1000,1000,1000))"))
print("\nMatrix Multiplication: [{}]*[{}] run [{}] iterations.".format("*5 ^2 ^2",str(multiplier),str(iterations)))

### CuPy and GPU
def run_gpu(x_gpu, multiplier):
    s = time.time()
    for x in range(1,multiplier):
        x_gpu *= 5
        x_gpu *= x_gpu
        x_gpu += x_gpu
        if x % math.ceil((multiplier/2)+0.1) == 0:
            os.system('nvidia-smi')
            print("Still running...")
    cp.cuda.Stream.null.synchronize()
    e = time.time()
    print("GPU: {}".format(e - s))

for y in range(1,6):
    run_gpu(x_gpu, multiplier)
    print("Completed cycle: [{}]".format(y))