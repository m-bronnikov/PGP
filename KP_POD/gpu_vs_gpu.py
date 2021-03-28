import matplotlib.pyplot as plt
import numpy as np

steps = [10*i for i in range(11)]

gpu16x16_times = np.array([
    9880.33, 7662.81, 6857.36, 13334, 10685.3, 11689,
    6958.75, 7003.95, 11684.1, 9622.73, 12843.4
])

gpu32x32_times = np.array([
    2805.09, 2254.31, 2017.94, 3814.61, 3118.57, 3398.04, 
    2068.66, 2077.44, 3199.37, 3074.54, 3855.83
])

gpu64x64_times = np.array([
    1158.1, 986.813, 873.69, 1726.75, 1188.78, 1372.67,
    931.751, 865.455, 1339.53, 1159.42, 1641.9
])

gpu128x128_times = np.array([
    1188.56, 918.489, 840.723, 1589.81, 1261.93, 1448.06,
    907.342, 881.653, 1361.58, 1187.7, 1561.19
])

plt.figure(figsize=(14, 9))
plt.title("Time of frames with 1228800 rays, 326 triangeles, 4 lights, depth 6")

plt.plot(steps, gpu16x16_times, label="GPU<<<16, 16>>>")
plt.plot(steps, gpu32x32_times, label="GPU<<<32, 32>>>")
plt.plot(steps, gpu64x64_times, label="GPU<<<64, 64>>>")
plt.plot(steps, gpu128x128_times, label="GPU<<<128, 128>>>")

plt.ylabel("time, ms")
plt.xlabel("frame number")
plt.legend()
plt.show()