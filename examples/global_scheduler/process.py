with open("/home/jovyan/hhy/VideoSys/examples/global_scheduler/logs.txt", 'r') as file:
    lines = file.readlines()
    start_time = float(lines[0].strip().split(' ')[-1])
    datas = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        else:
            data = line.strip().split(' ')
            id = int(line[1])
            end_time = float(line[4])
            resolution = str(line[-1])  
            datas.append((id,end_time-start_time,resolution))
    datas.sort(key=lambda x:x[0])
    for data in datas:
        print(data[0])
    for data in datas:
        print(data[1])
    for data in datas:
        print(data[2])