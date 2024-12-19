schedule_policies = ["Cluster_Isolated", "Round_Robin", "Best_Match", "Best_Match_Dynamic", "Step_Wise_SP"]
prefix_path = "/home/jovyan/hhy/VideoSys/examples/global_scheduler/logs_"
suffix_path = ".txt"
ratios = [(1,1,8),(2,2,6),(3,3,4),(1,8,1),(2,6,2),(3,4,3),(8,1,1),(6,2,2),(4,3,3),(2,4,4),(4,2,4),(4,4,2),(1,3,6),(6,1,3),(3,6,1)]

#count = 0
for x, y, z in ratios:
    for policy in schedule_policies:
        file_path = prefix_path + str(x) + "_" + str(y) + "_" + str(z) + "_" + policy + suffix_path
        print(f"----------Ratio----------")
        print(f"144p: 240p: 360p = {x}: {y}: {z}")
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                start_time = float(lines[0].strip().split(' ')[-1])
                datas = []
                res = {}
                for i, line in enumerate(lines):
                    if i == 0:
                        continue
                    else:
                        data = line.strip().split(' ')
                        id = int(data[1])
                        end_time = float(data[4])
                        resolution = str(data[-1])  
                        datas.append((id,end_time-start_time,resolution))
                        if resolution not in res:
                            res[resolution] = 1
                        else:
                            res[resolution] += 1
                datas.sort(key=lambda x:x[0])
                '''if count == 0:
                    print(f"----------IDs----------")
                    for data in datas:
                        print(data[0])
                    print(f"----------Resolutions----------")
                    for data in datas:
                        print(data[2])'''
                print(f"----------Check----------")
                for key, value in res.items():
                    print(f"{key} appears {value} times")
                    #count += 1
                print(f"----------{policy} Durations----------")
                avg_time = sum([data for _, data, _ in datas]) / len([data for _, data, _ in datas])
                print(avg_time)
        except TypeError:
            print(f"The {file_path} is none")
        except FileNotFoundError:
            print(f"The {file_path} does not exist")
        except Exception as e:
            print(f"Unexpected error {e}")