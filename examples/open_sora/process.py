import os
import numpy as np

def main():
    start_log_path = "/workspace/VideoSys/examples/open_sora/start_log.txt"
    dit_log_path = "/workspace/VideoSys/examples/open_sora/dit_log.txt"
    vae_log_path = "/workspace/VideoSys/examples/open_sora/vae_log.txt"

    slo_720p_5 = 118.9662282
    slo_720p_10 = 237.9324563
    slo_360p_5 = 31.24583801
    slo_360p_10 = 62.49167601
    slo_480p_5 = 51.75390204
    slo_480p_10 = 103.5078041
    video_360p = 2.089775626
    video_720p = 10.40768774
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

    with open(dit_log_path, 'r') as file:
        lines = file.readlines()
        processes = {}
        reslo = {}
        ends = {}  
        for line in lines:
            datas = line.strip().split(' ')
            req_id = str(datas[1])
            end_time = float(datas[-1])
            process_time = float(datas[-5])
            resolution = str(datas[3])
            if req_id not in processes:
                processes[req_id] = process_time
            if req_id not in reslo:
                reslo[req_id] = resolution
            if req_id not in ends:
                ends[req_id] = end_time

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
        if key in processes:
            print(processes[key])
    print(f"----------Ends----------")
    for key in starts.keys():
        if key in ends:
            print(ends[key])
    print(f"----------Resolutions----------")
    for key in starts.keys():
        if key in reslo:
            print(reslo[key])
        
    outputs = []
    slo5 = 0
    slo10 = 0
    for key, value in starts.items():
        if key not in ends or key not in reslo:
            continue
        cur_res = reslo[key]
        if cur_res == "360p":
            ends[key] = ends[key] + video_360p
        if cur_res == "720p":
            ends[key] = ends[key] + video_720p
        outputs.append(ends[key] - value)
        
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
        if cur_res == "480p":
            if outputs[-1] <= slo_480p_5:
                slo5 += 1
            if outputs[-1] <= slo_480p_10:
                slo10 += 1
    final_outputs = np.array(outputs)
    print(f"----------AVG----------")
    print(np.mean(final_outputs))
    print(f"----------P50----------")
    print(np.percentile(final_outputs, 50, method = 'higher'))
    print(f"----------P90----------")
    print(np.percentile(final_outputs, 90, method = 'higher'))
    print(f"----------P99----------")
    print(np.percentile(final_outputs, 99, method = 'higher'))
    print(f"----------SLO5----------")
    print(slo5 / len(outputs))
    print(f"----------SLO10----------")
    print(slo10 / len(outputs))
    
    start_log_path2 = "/workspace/VideoSys/metrics/new_ljf_start.txt"
    dit_log_path2 = "/workspace/VideoSys/metrics/new_ljf_dit.txt"
    vae_log_path2 = "/workspace/VideoSys/metrics/new_ljf_vae.txt"
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
        if os.path.exists(dit_log_path):
            os.rename(dit_log_path, dit_log_path2)
            print(f"文件已成功从 '{dit_log_path}' 移动到 '{dit_log_path2}'。")
        else:
            print(f"错误：源文件 '{dit_log_path}' 不存在。")
    except FileExistsError:
        # 如果目标文件已存在且os.rename无法覆盖，会抛出此错误
        print(f"错误：目标文件 '{dit_log_path2}' 已存在。")
    except OSError as e:
        # 处理其他可能的操作系统错误，如权限不足
        print(f"发生操作系统错误：{e}")
    
    try:
        # 检查源文件是否存在
        if os.path.exists(vae_log_path):
            os.rename(vae_log_path, vae_log_path2)
            print(f"文件已成功从 '{vae_log_path}' 移动到 '{vae_log_path2}'。")
        else:
            print(f"错误：源文件 '{vae_log_path}' 不存在。")
    except FileExistsError:
        # 如果目标文件已存在且os.rename无法覆盖，会抛出此错误
        print(f"错误：目标文件 '{vae_log_path2}' 已存在。")
    except OSError as e:
        # 处理其他可能的操作系统错误，如权限不足
        print(f"发生操作系统错误：{e}")

if __name__ == "__main__":
    main()