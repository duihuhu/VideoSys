prefix_path = "/home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity2/"
prefix_path2 = "/home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/logs_temp/"
prefix_path3 = "/home/jovyan/hhy/logs/"
ratios = [(2,2,6),(2,6,2),(6,2,2),(2,4,4),(4,2,4),(4,4,2),(1,3,6),(6,1,3),(3,6,1),(1,1,1)]


ddits = []
ddits2 = []
statics = []
statics2 = []
for j, cls in enumerate(["ddit.txt", "static.txt"]):
    #if j == 1:
    #    break
    for x, y, z in ratios:
        file_path = prefix_path + str(x) + "_" + str(y) + "_" + str(z) + "_" + cls
        times = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    st = float(line.strip().split(' ')[-1])
                else:
                    times.append(float(line.strip().split(' ')[-1]) - st)
        if j == 0:
            ddits.append(sum(times) / len(times))
            ddits2.append(max(times))
        else:
            statics.append(sum(times) / len(times))
            statics2.append(max(times))
print(f"----------Avg----------")
print(f"----------DDiT----------")
for item in ddits:
    print(item)
print(f"----------Static----------")
for item in statics:
    print(item)
print(f"----------Tail----------")
print(f"----------DDiT----------")
for item in ddits2:
    print(item)
print(f"----------Static----------")
for item in statics2:
    print(item)

'''for j, cls in enumerate(["ddit.txt", "static.txt"]):
    file_path = prefix_path2 + cls
    times = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                st = float(line.strip().split(' ')[-1])
            else:
                times.append(float(line.strip().split(' ')[-1]) - st)
    if j == 0:
        ddits.append(sum(times) / len(times))
        ddits.append(max(times))
    else:
        statics.append(sum(times) / len(times))
        statics.append(max(times))
print(f"----------DDiT----------")
for item in ddits:
    print(item)
print(f"----------Static----------")
for item in statics:
    print(item)'''