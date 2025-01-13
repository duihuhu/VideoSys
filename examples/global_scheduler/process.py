prefix_path = "/home/jovyan/hhy/VideoSys/examples/global_scheduler/logs_"
suffix_path = ".txt"
ratios = [(1,1,8),(2,2,6),(3,3,4),(1,8,1),(2,6,2),(3,4,3),(8,1,1),(6,2,2),(4,3,3),(2,4,4),(4,2,4),(4,4,2),(1,3,6),(6,1,3),(3,6,1),(1,1,1)]

for x, y, z in ratios:
    print(f"----------Ratio----------")
    print(f"144p: 240p: 360p = {x}: {y}: {z}")
    file_path = prefix_path + str(x) + "_" + str(y) + "_" + str(z) + suffix_path
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip()
            print(data)