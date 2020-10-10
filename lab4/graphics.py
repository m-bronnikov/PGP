import matplotlib.pyplot as plt

PATH = "logs.log"

modes = {}

with open(PATH, 'r') as fin:
    while True:
        title = fin.readline()
        if not title:
            break
        treads = int(fin.readline())
        size = int(fin.readline())
        el_time = float(fin.readline())
        if title not in modes:
            modes[title] = ([], [])
        modes[title][0].append(treads)
        modes[title][1].append(el_time)


fig = plt.figure(figsize=(8, 15))


plt.title("Времени затрачено на разложение матрицы 3840*3840")
for key in modes.keys():
    pair = modes[key]
    plt.plot(pair[0], pair[1], label=key)


plt.grid()
plt.xlabel("Количество потоков на блок(равно количеству блоков)")
plt.ylabel("Время, мс")

plt.legend()
plt.show()

#fig.savefig('temp.png', dpi=fig.dpi)

