import matplotlib.pyplot as plt
import numpy as np

gpu3_times = np.array([
    380.282, 245.127, 232.295, 403.807, 323.945, 363.7, 
    230.198, 252.602, 348.558, 300.52, 409.121
]).mean()
gpu2_times = np.array([
    220.809, 121.409, 115.017, 197.234, 162.063, 172.602,
    118.744, 115.917, 171.972, 146.747, 193.799
]).mean()
gpu1_times = np.array([
    78.809, 61.586, 60.791, 85.615, 69.837, 78.786,
    60.53, 60.15, 81.404, 70.469, 88.563
]).mean()

plt.figure(figsize=(14, 9))
plt.title("Mean time of rendering with 1228800 rays 4 lights, depth 6")
plt.plot([78, 170, 326], [gpu1_times, gpu2_times, gpu3_times], label="GPU<<<64, 64>>>")

plt.xlabel("Triangles")
plt.ylabel("time, ms")
plt.legend()
plt.show()