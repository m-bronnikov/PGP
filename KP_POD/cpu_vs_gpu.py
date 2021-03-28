import matplotlib.pyplot as plt
import numpy as np

cpu_times = np.array([
    84210.5, 63232.6, 60347.2, 119125, 85544.7, 103039, 
    60775.7, 59175.2, 99285.5, 82559.6, 113267
]).mean()
gpu_times = np.array([
    1075.28, 905.792, 794.473, 1493.21, 1177.44, 1276.6,
    814.819, 802.823, 1277.62, 1070.7, 1452.65
]).mean()

plt.figure(figsize=(14, 9))
plt.title("Mean Time of each frame with 1228800 rays, 326 triangeles, 4 lights, depth 6")
plt.bar("CPU", cpu_times, label="CPU")
plt.bar("GPU", gpu_times, label="GPU<<<256, 256>>")
plt.yticks(np.arange(0, 86000, 2000))
plt.ylabel("time, ms")
plt.legend()
plt.show()