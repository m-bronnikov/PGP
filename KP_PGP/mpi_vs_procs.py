import numpy as np
import matplotlib.pyplot as plt

proc8 = [1, 2, 3, 4, 5, 8]
mpi8 = [515.958, 488.203, 470.14, 430.308, 477.727, 481.197]
proc1 = [1, 2, 4, 8]
mpi1 = [1366.324, 919.216, 609.055, 564.163]

plt.title("Time of rendering 24 frames")
plt.plot(proc8, mpi8, label="8 Threads OMP")
plt.plot(proc1, mpi1, label="1 Thread OMP")
plt.xlabel("Procs")
plt.ylabel("tims, s")
plt.legend()
plt.show()