'''prefix_path = "/home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity2/"
prefix_path2 = "/home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/logs_temp/"
prefix_path3 = "/home/jovyan/hhy/logs/"
ratios = [(1,1,8),(2,2,6),(3,3,4),(1,8,1),(2,6,2),(3,4,3),(8,1,1),(6,2,2),(4,3,3),(2,4,4),(4,2,4),(4,4,2),(1,3,6),(6,1,3),(3,6,1),(1,1,1)]
ddits = []
ddits2 = []
statics = []
statics2 = []
for j, cls in enumerate(["ddit.txt", "static.txt"]):
    if j == 1:
        break
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
    print(item)'''

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

'''dit_log_path = "/workspace/VideoSys/examples/global_scheduler/dit_log.txt"
vae_log_path = "/workspace/VideoSys/examples/open_sora/vae_log.txt"

with open(dit_log_path, 'r') as file:
    lines = file.readlines()
    dits = []
    for line in lines[5:]:
        datas = line.strip().split(' ')
        costs = float(datas[-1])
        dits.append(costs)
    print(f"----------DiT----------")
    print(sum(dits) / len(dits))

with open(vae_log_path, 'r') as file:
    lines = file.readlines()
    vaes = []
    for line in lines[5:]:
        datas = line.strip().split(' ')
        costs = float(datas[-1])
        vaes.append(costs)
    print(f"----------Vae----------")
    print(sum(vaes) / len(vaes))'''

import argparse
import os
import numpy as np

def main(dop: int):
    start_log_path = "/workspace/VideoSys/examples/global_scheduler/start_log.txt"
    end_log_path = "/workspace/VideoSys/examples/open_sora/end_log.txt"
    #dit_log_path = "/workspace/VideoSys/examples/global_scheduler/dit_log.txt"
    #vae_log_path = "/workspace/VideoSys/examples/open_sora/vae_log.txt"

    slo_720p_5 = 118.9662282
    slo_720p_10 = 237.9324563
    slo_360p_5 = 31.24583801
    slo_360p_10 = 62.49167601
    with open(start_log_path, 'r') as file:
        lines = file.readlines()
        starts = {}
        for line in lines:
            datas = line.strip().split(' ')
            req_id = str(datas[1])
            start_time = float(datas[-1])
            if req_id not in starts:
                starts[req_id] = start_time
    print(f"----------Starts----------")
    for value in starts.values():
        print(value)

    with open(end_log_path, 'r') as file:
    #with open(start_log_path, 'r') as file:
        lines = file.readlines()
        ends = {}
        processes = {}
        reslo = {}
        #ends = []    
        for line in lines:
            datas = line.strip().split(' ')
            req_id = str(datas[1])
            end_time = float(datas[-1])
            process_time = float(datas[-4])
            resolution = str(datas[3])
            if req_id not in ends:
                ends[req_id] = end_time
            if req_id not in processes:
                processes[req_id] = process_time
            if req_id not in reslo:
                reslo[req_id] = resolution
            #ends.append(end_time)
    '''with open(vae_log_path, 'r') as file:
        lines = file.readlines()
        ends = {}
        for line in lines:
            datas = line.strip().split(' ')
            req_id = str(datas[1])
            end_time = float(datas[-1])
            if req_id not in ends:
                ends[req_id] = end_time
    '''
    print(f"----------Processes----------")
    for key in starts.keys():
        print(processes[key])
    print(f"----------Ends----------")
    for key in starts.keys():
        print(ends[key])
    print(f"----------Resolutions----------")
    for key in starts.keys():
        print(reslo[key])
        
    outputs = []
    slo5 = 0
    slo10 = 0
    for key, value in starts.items():
        outputs.append(ends[key] - value)
        cur_res = reslo[key]
        if cur_res == "720p":
            if outputs[-1] <= slo_720p_5:
                slo5 += 1
            if outputs[-1] <= slo_720p_10:
                slo10 += 1
        if cur_res == "360p":
            if outputs[-1] <= slo_360p_5:
                slo5 += 1
            if outputs[-1] <= slo_360p_10:
                slo10 += 1
    final_outputs = np.array(outputs)
    print(f"----------AVG----------")
    print(np.mean(final_outputs))
    print(f"----------P50----------")
    print(np.median(final_outputs))
    print(f"----------P90----------")
    print(np.percentile(final_outputs, 90))
    print(f"----------P99----------")
    print(np.percentile(final_outputs, 99))
    print(f"----------SLO5----------")
    print(slo5 / len(outputs))
    print(f"----------SLO10----------")
    print(slo10 / len(outputs))
    #print(f"----------Avg----------")
    #print(sum(ends[3:]) / len(ends[3:]))
    

    start_log_path2 = "/workspace/VideoSys/metrics/new_naive_start_" + str(dop) + ".txt"
    end_log_path2 = "/workspace/VideoSys/metrics/new_naive_end_" + str(dop) + ".txt"
    try:
        # 检查源文件是否存在
        if os.path.exists(start_log_path):
            os.rename(start_log_path, start_log_path2)
            print(f"文件已成功从 '{start_log_path}' 移动到 '{start_log_path2}'。")
        else:
            print(f"错误：源文件 '{start_log_path}' 不存在。")
    except FileExistsError:
        # 如果目标文件已存在且os.rename无法覆盖，会抛出此错误
        print(f"错误：目标文件 '{start_log_path2}' 已存在。")
    except OSError as e:
        # 处理其他可能的操作系统错误，如权限不足
        print(f"发生操作系统错误：{e}")
    
    try:
        # 检查源文件是否存在
        if os.path.exists(end_log_path):
            os.rename(end_log_path, end_log_path2)
            print(f"文件已成功从 '{end_log_path}' 移动到 '{end_log_path2}'。")
        else:
            print(f"错误：源文件 '{end_log_path}' 不存在。")
    except FileExistsError:
        # 如果目标文件已存在且os.rename无法覆盖，会抛出此错误
        print(f"错误：目标文件 '{end_log_path2}' 已存在。")
    except OSError as e:
        # 处理其他可能的操作系统错误，如权限不足
        print(f"发生操作系统错误：{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dop", type=int, default=8)
    args = parser.parse_args()
    main(args.dop)

'''root_path = "/home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/mock_stream/"
ratios = [(2,2,6),(2,6,2),(6,2,2),(2,4,4),(4,2,4),(4,4,2),(1,3,6),(6,1,3),(3,6,1),(1,1,1)]
recv_ratio = [0.25, 0.5, 0.75, 1]
types = [0, 1, 2, 3, 4]
for x, y, z in ratios:
    for rr in recv_ratio:
        rr_avgs = []
        rr_tails = []
        for t in types:
            start_log_path = root_path + str(x) + "_" + str(y) + "_" + str(z) + "_" + str(rr) + "_" + str(t) + "_start.txt"
            end_log_path = root_path + str(x) + "_" + str(y) + "_" + str(z) + "_" + str(rr) + "_" + str(t) + "_end.txt"
            starts = {}
            ends = {}
            with open(start_log_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    datas = line.strip().split(' ')
                    req_id = str(datas[1])
                    start_time = float(datas[-1])
                    if req_id not in starts:
                        starts[req_id] = start_time
            with open(end_log_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    datas = line.strip().split(' ')
                    req_id = str(datas[1])
                    end_time = float(datas[-1])
                    if req_id not in ends:
                        ends[req_id] = end_time
            outputs = []
            for key, value in starts.items():
                outputs.append(ends[key] - value)
            rr_avgs.append(sum(outputs) / len(outputs))
            rr_tails.append(max(outputs))
        print(f"----------{x}_{y}_{z}_{t}----------")
        print(f"----------Avg----------")
        for item in rr_avgs:
            print(item)
        print(f"----------Tail----------")
        for item in rr_tails:
            print(item)'''
'''starts = {}
ends = {}
with open(start_log_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        datas = line.strip().split(' ')
        req_id = str(datas[1])
        start_time = float(datas[-1])
        if req_id not in starts:
            starts[req_id] = start_time
with open(end_log_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        datas = line.strip().split(' ')
        req_id = str(datas[1])
        end_time = float(datas[-1])
        if req_id not in ends:
            ends[req_id] = end_time
#start_times = list(starts.values())
#start_times.sort()
outputs = []
for key, value in starts.items():
    outputs.append(ends[key] - value)
#for _, value in ends.items():
#    outputs.append(value - start_times[0])
print(f"----------Avg----------")
print(sum(outputs) / len(outputs))
print(f"----------Tail----------")
print(max(outputs))
outputs.sort(key = lambda x: x)
for item in outputs:
    print(item)

file_path = "/data/home/scyb091/VideoSys/examples/global_scheduler/log.txt"
with open(file_path, 'r') as file:
    dits = []
    vaes = []
    lines = file.readlines()
    for i, line in enumerate(lines):
        if i < 8: # 4 2 1
            continue
        data = line.strip().split(' ')
        dits.append(float(data[-2]))
        vaes.append(float(data[-1]))
    print(f"----------DiT----------")
    print(sum(dits) / len(dits))
    print(f"----------Vae----------")
    print(sum(vaes) / len(vaes))'''