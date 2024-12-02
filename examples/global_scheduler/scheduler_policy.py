import random
import threading
import queue
import os
import time
#global config
resolutions = ["144p", "240p", "360p"]
resolutions_weights = [6, 3, 1]
dit_config = {"144p":3, "240p": 4.6, "360p": 6.1}
vae_config = {"144p":0.16, "240p": 0.38, "360p": 0.87}

dit_configs = {"144p":{1:3, 2:3.4, 4:3.5}, "240p":{1:8.3, 2:4.6, 4:3.7}, "360p":{1:19.2, 2:10.4, 4:6.1}}
vae_configs = {"144p":{1:0.16, 2:0.16, 4:0.16}, "240p":{1:0.38, 2:0.38, 4:0.38}, "360p":{1:0.87, 2:0.87, 4:0.87}}

opt_configs = {"144p":1, "240p": 2, "360p": 4}


# 初始化锁
gpu_status_lock = threading.Lock()

class Task:
    def __init__(self, index, resolution) -> None:
        self.index = index
        self.resolution = resolution

def mock_workload(num_tasks):
    tasks= []
    for index in range(num_tasks):
        resolution= random.choices(resolutions, weights=resolutions_weights, k=1)[0]
        task = Task(index=index, resolution=resolution)
        tasks.append(task)
    return tasks

def execute_isolated_cluster():
    while True:
        task = task_queue.get()  # 阻塞，直到有任务
        if task == "STOP":
            print(f"Worker {os.getpid()} stopping.")
            break
        else:
            t1 = time.time()
            print("start execute task time ", t1)
            dit_time = dit_config[task.resolution]
            vae_time = vae_config[task.resolution]
            gpus = gpu_queue[task.resolution]
            gpus.get()

            time.sleep(dit_time)
            time.sleep(vae_time)
            gpus.put(0)
            t2 = time.time()
            print("end execute task time ", task.resolution, t2-t1, t2)


def match_best_gpu(num_gpu):
    global gpu_status_lock
    with gpu_status_lock:
        available_indices = [i for i, status in enumerate(gpu_status) if status == 0]
        
        if not available_indices:
            return None, None # 如果没有可用的 GPU
        
        best_index = min(available_indices, key=lambda i: abs(gpu_nums[i] - num_gpu))
        gpu_status[best_index] = 1
    # 返回最佳 GPU 的位置
    return gpu_nums[best_index], best_index

def execute_mixed_cluster_static_awareness():
    while True:
        task = task_queue.get()  # 阻塞，直到有任务
        if task == "STOP":
            print(f"Worker {os.getpid()} stopping.")
            break
        else:
            t1 = time.time()
            print("start execute task time ", t1)
            matched_gpu = None
            while matched_gpu == None:
                num_gpu = opt_configs[task.resolution]
                matched_gpu, index = match_best_gpu(num_gpu)
            dit_time = dit_configs[task.resolution][matched_gpu]
            vae_time = vae_configs[task.resolution][matched_gpu]
            #mock
            time.sleep(dit_time)
            time.sleep(vae_time)
            gpu_status[index] = 0
            t2 = time.time()
            print("end execute task time ", task.resolution, matched_gpu, t2-t1, t2)

def execute_mixed_cluster_round_robin():
    while True:
        task = task_queue.get()  # 阻塞，直到有任务
        if task == "STOP":
            print(f"Worker {os.getpid()} stopping.")
            break
        else:
            t1 = time.time()
            print("start execute task time ", t1)
            gpu_num = gpu_queue.get()
            dit_time = dit_configs[task.resolution][gpu_num]
            vae_time = vae_configs[task.resolution][gpu_num]
            #mock
            time.sleep(dit_time)
            time.sleep(vae_time)
            gpu_queue.put(gpu_num)
            t2 = time.time()
            print("end execute task time ", task.resolution, gpu_num, t2-t1, t2)

#if use slo: for choose gpu num
#no use slo: only dynamic deployment to awareness resolution
def execute_mixed_cluster_dynamic_awareness():
    return
   
if __name__ == "__main__":
    n = 2
    num_gpu = (1 + 2 + 4) * 2
    gpu_queue = {"144p": queue.Queue(), "240p": queue.Queue(), "360p":  queue.Queue()}
    for k, v in gpu_queue.items():
        v.put(0)
        v.put(0)
        
    task_queue = queue.Queue()
    consumers = []
    num_tasks = 10
    tasks = mock_workload(num_tasks=num_tasks)
    for i in range(n*len(resolutions)):
        consumer = threading.Thread(target=execute_isolated_cluster)
        consumer.start()
        consumers.append(consumer)
        
    for task in tasks:
        task_queue.put(task)
    
    for i in range(n*len(resolutions)):
        task_queue.put("STOP")
    
    for consumer in consumers:
        consumer.join()
        
    print("-------------------------------------")
    gpu_list = [1, 1, 2, 2, 4, 4]
    random.shuffle(gpu_list)
    gpu_queue = queue.Queue()
    for gpu_num in gpu_list:
        gpu_queue.put(gpu_num)

    for i in range(n*len(resolutions)):
        consumer = threading.Thread(target=execute_mixed_cluster_round_robin)
        consumer.start()
        consumers.append(consumer)
        
    for task in tasks:
        task_queue.put(task)
    
    for i in range(n*len(resolutions)):
        task_queue.put("STOP")
        
    for consumer in consumers:
        consumer.join()
    print("-------------------------------------")
    
    gpu_nums = [1, 1, 2, 2, 4, 4]
    gpu_status = [0, 0, 0, 0, 0]

    for i in range(n*len(resolutions)):
        consumer = threading.Thread(target=execute_mixed_cluster_static_awareness)
        consumer.start()
        consumers.append(consumer)

    for task in tasks:
        task_queue.put(task)
    
    for i in range(n*len(resolutions)):
        task_queue.put("STOP")
        
    for consumer in consumers:
        consumer.join()