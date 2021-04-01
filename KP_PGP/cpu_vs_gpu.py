import matplotlib.pyplot as plt
import numpy as np

omp_times = np.array([
    28592.5, 25641.5, 23193.3, 21877.1, 22362.7, 22501, 
    20734.5, 22472.9, 24743.3, 23363.7, 23813.5, 24189
]).mean()
cpu_times = np.array([
    84210.5, 63232.6, 60347.2, 119125, 85544.7, 103039, 
    60775.7, 59175.2, 99285.5, 82559.6, 113267
]).mean()
gpu_times = np.array([
    1050.2, 1026.7, 1045.34, 1104.25, 1287.18, 1282.78, 
    1312.19, 1635.34, 1830.13, 1792.39, 2047.93, 2107.92, 1837.97
]).mean()

print(gpu_times)

plt.figure(figsize=(14, 9))
plt.title("Mean Time of each frame with 1228800 rays, 326 triangeles, 4 lights, depth 6")
plt.bar("OMP", omp_times, label="OMP(8 threads)")
plt.bar("CPU", cpu_times, label="CPU")
plt.bar("GPU", gpu_times, label="GPU<<<256, 256>>")
plt.yticks(np.arange(0, 86000, 2000))
plt.ylabel("time, ms")
plt.legend()
plt.show()