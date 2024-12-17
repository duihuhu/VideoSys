schedule_policies = ["Cluster_Isolated", "Round_Robin", "Best_Match"]
prefix_path = "/home/jovyan/hhy/VideoSys/examples/global_scheduler/logs_"
suffix_path = ".txt"

for i, policy in enumerate(schedule_policies):
    file_path = prefix_path + policy + suffix_path
    print(f"----------{policy}----------")
    with open(file_path, 'r') as file:
        lines = file.readlines()
        start_time = float(lines[0].strip().split(' ')[-1])
        datas = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            else:
                data = line.strip().split(' ')
                id = int(data[1])
                end_time = float(data[4])
                resolution = str(data[-1])  
                datas.append((id,end_time-start_time,resolution))
        datas.sort(key=lambda x:x[0])
        if i == 0:
            print(f"----------IDs----------")
            for data in datas:
                print(data[0])
            print(f"----------Resolutions----------")
            for data in datas:
                print(data[2])
        print(f"----------Durations----------")
        for data in datas:
            print(data[1])