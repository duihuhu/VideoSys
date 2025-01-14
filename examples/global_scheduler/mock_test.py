from typing import Dict, Deque, List, Tuple, Optional
from queue import Queue
import copy
import time
import threading
import argparse
import random
import sys
import os

class Request:
    def __init__(self, id: int, resolution: str):
        self.id = id
        self.resolution = resolution
        self.workers_ids: List[int] = []
        self.workers_ids2: List[Tuple[int, int]] = []

finished_requests: List[int] = []
requests_new_workers_ids: Dict[int, List[int]] = {}
requests_new_workers_ids2: Dict[int, List[Tuple[int, int]]] = {}
requests_cur_steps: Dict[int, int] = {}

if sys.version_info >= (3, 9):
    tasks_queue: Queue[Request] = Queue()
else:
    tasks_queue: Queue = Queue()

class GlobalScheduler:
    def __init__(self, instances_num: int, jobs_num: int, high_affinity: bool = True, gpus_per_instance: int = 8):
        self.gpu_status = [0 for _ in range(instances_num * gpus_per_instance)]
        self.hungry_requests: Dict[int, Request] = {}
        self.waiting_requests: Deque[Request] = Deque()
        self.requests_workers_ids: Dict[int, List[int]] = {}
        self.requests_workers_ids2: Dict[int, List[Tuple[int, int]]] = {}
        self.requests_last_steps: Dict[int, int] = {}
        self.dit_times: Dict[str, Dict[int, float]] = {"144p": {1: 3, 2: 3.4, 4: 3.5}, 
                                                       "240p": {1: 8.3, 2: 4.6, 4: 3.7}, 
                                                       "360p": {1: 19.2, 2: 10.4, 4: 6.1}}
        self.opt_gpus_num: Dict[str, int] = {"144p": 1, "240p": 2, "360p": 4}
        self.denoising_steps: int = 30
        self.jobs_num = jobs_num
        self.high_affinity = high_affinity
        self.gpu_status2 = [[0 for _ in range(gpus_per_instance)] for _ in range(instances_num)]

    
    def add_request(self, request: Request) -> None:
        self.waiting_requests.append(request)
    
    def update_gpu_status_static(self, request_id: int) -> None:
        if self.high_affinity:
            for gpu_id in self.requests_workers_ids[request_id]:
                self.gpu_status[gpu_id] = 0
        else:
            for x, y in self.requests_workers_ids2[request_id]:
                self.gpu_status2[x][y] = 0
    
    def update_gpu_status(self, last: bool, request_id: int) -> None:
        if self.high_affinity:
            if last:
                self.gpu_status[self.requests_workers_ids[request_id][0]] = 0
                self.requests_workers_ids.pop(request_id, None)
            else:
                for i in range(1, len(self.requests_workers_ids[request_id])):
                    self.gpu_status[self.requests_workers_ids[request_id][i]] = 0
                self.hungry_requests.pop(request_id, None)
                requests_cur_steps.pop(request_id, None)
                self.requests_last_steps.pop(request_id, None)
        else:
            if last:
                gpu_id_row, gpu_id_column = self.requests_workers_ids2[request_id][0]
                self.gpu_status2[gpu_id_row][gpu_id_column] = 0
                self.requests_workers_ids2.pop(request_id, None)
            else:
                for i in range(1, len(self.requests_workers_ids2[request_id])):
                    x, y = self.requests_workers_ids2[request_id][i]
                    self.gpu_status2[x][y] = 0
                self.hungry_requests.pop(request_id, None)
                requests_cur_steps.pop(request_id, None)
                self.requests_last_steps.pop(request_id, None)
    
    def get_free_gpus_topology(self) -> List[List[Tuple[int, int]]]:
        output = []
        for i in range(len(self.gpu_status2)):
            temp = []
            for j in range(len(self.gpu_status2[i])):
                if self.gpu_status2[i][j] == 0:
                    temp.append((i, j))
            output.append(temp)
        return output

    def affinity_aware_hungry_first_priority_schedule(self) -> Request:
        if self.high_affinity:
            cur_free_gpus: Queue[int] = Queue()
            for gpu_id, status in enumerate(self.gpu_status):
                if status == 0:
                    cur_free_gpus.put(gpu_id)
            if cur_free_gpus.qsize() < 1:
                return None
        else:
            cur_free_gpus2 = self.get_free_gpus_topology()
            cur_free_gpus2.sort(key = lambda x: len(x), reverse = True)
            if len(cur_free_gpus2[0]) < 1:
                return None
        #----------process hungry queue in starvation descending order while num = N----------#
        temp_hungry_requests = list(self.hungry_requests.values())
        # sort in descending order by starvation time
        if self.high_affinity:
            temp_hungry_requests.sort(key = lambda x: (requests_cur_steps[x.id] - self.requests_last_steps[x.id])
                                        * (self.dit_times[x.resolution][len(self.requests_workers_ids[x.id])] - self.dit_times[x.resolution][self.opt_gpus_num[x.resolution]])
                                        / self.denoising_steps
                                        , reverse = True)
        else:
            temp_hungry_requests.sort(key = lambda x: (requests_cur_steps[x.id] - self.requests_last_steps[x.id])
                                        * (self.dit_times[x.resolution][len(self.requests_workers_ids2[x.id])] - self.dit_times[x.resolution][self.opt_gpus_num[x.resolution]])
                                        / self.denoising_steps
                                        , reverse = True)
        for cur_hungry_request in temp_hungry_requests:
            if self.high_affinity:
                if cur_free_gpus.qsize() < 1:
                    break
            else:
                if len(cur_free_gpus2[0]) < 1:
                    break
            cur_wanted_gpus_num = []
            cur_opt_gpus_num = self.opt_gpus_num[cur_hungry_request.resolution]
            while cur_opt_gpus_num > 0:
                if self.high_affinity:
                    gap_gpus_num = cur_opt_gpus_num - len(self.requests_workers_ids[cur_hungry_request.id])
                else:
                    gap_gpus_num = cur_opt_gpus_num - len(self.requests_workers_ids2[cur_hungry_request.id])
                if gap_gpus_num > 0:
                    cur_wanted_gpus_num.append(gap_gpus_num)
                cur_opt_gpus_num //= 2
            for i, wanted_gpus_num in enumerate(cur_wanted_gpus_num):
                if self.high_affinity:
                    if cur_free_gpus.qsize() < wanted_gpus_num:
                        continue
                else:
                    if len(cur_free_gpus2[0]) < wanted_gpus_num:
                        continue
                for _ in range(wanted_gpus_num):
                    if self.high_affinity:
                        gpu_id = cur_free_gpus.get()
                        self.gpu_status[gpu_id] = 1
                        self.requests_workers_ids[cur_hungry_request.id].append(gpu_id)
                    else:
                        gpu_id_row, gpu_id_column = cur_free_gpus2[0].pop(0) # update the max row itself
                        #cur_free_gpus2[0][0] -= 1 # update free gpu num in the max row
                        self.gpu_status2[gpu_id_row][gpu_id_column] = 1
                        self.requests_workers_ids2[cur_hungry_request.id].append((gpu_id_row, gpu_id_column))
                # sort again in case the max row not be the max
                if not self.high_affinity:
                    cur_free_gpus2.sort(key = lambda x: len(x), reverse = True)
                # notice the real workers
                if self.high_affinity:
                    requests_new_workers_ids[cur_hungry_request.id] = copy.deepcopy(self.requests_workers_ids[cur_hungry_request.id])
                else:
                    requests_new_workers_ids2[cur_hungry_request.id] = copy.deepcopy(self.requests_workers_ids2[cur_hungry_request.id])
                if i == 0:
                    self.hungry_requests.pop(cur_hungry_request.id, None)
                    requests_cur_steps.pop(cur_hungry_request.id, None)
                    self.requests_last_steps.pop(cur_hungry_request.id, None)
                else:
                    self.requests_last_steps[cur_hungry_request.id] = requests_cur_steps[cur_hungry_request.id]     
        #----------process waiting queue in FCFS while num = 1----------#
        if self.waiting_requests:
            if self.high_affinity:
                if cur_free_gpus.qsize() < 1:
                    return None
            else:
                if len(cur_free_gpus2[0]) < 1:
                    return None
            cur_waiting_request = self.waiting_requests[0]
            # help to end the first loop
            if cur_waiting_request == "exit":
                self.waiting_requests.popleft()
                return cur_waiting_request
            
            cur_demand_gpus_num = []
            cur_max_gpus_num = self.opt_gpus_num[cur_waiting_request.resolution]
            while cur_max_gpus_num > 0:
                cur_demand_gpus_num.append(cur_max_gpus_num)
                cur_max_gpus_num //= 2
            for j, demand_gpus_num in enumerate(cur_demand_gpus_num):
                if self.high_affinity:
                    if cur_free_gpus.qsize() < demand_gpus_num:
                        continue
                else:
                    if len(cur_free_gpus2[0]) < demand_gpus_num:
                        continue
                for _ in range(demand_gpus_num):
                    if self.high_affinity:
                        gpu_id = cur_free_gpus.get()
                        self.gpu_status[gpu_id] = 1
                        if cur_waiting_request.id not in self.requests_workers_ids:
                            self.requests_workers_ids[cur_waiting_request.id] = [gpu_id]
                        else:
                            self.requests_workers_ids[cur_waiting_request.id].append(gpu_id)
                    else:
                        gpu_id_row, gpu_id_column = cur_free_gpus2[0].pop(0) # update the max row itself
                        #cur_free_gpus2[0][0] -= 1 # update free gpu num in the max row
                        self.gpu_status2[gpu_id_row][gpu_id_column] = 1
                        if cur_waiting_request.id not in self.requests_workers_ids2:
                            self.requests_workers_ids2[cur_waiting_request.id] = [(gpu_id_row, gpu_id_column)]
                        else:
                            self.requests_workers_ids2[cur_waiting_request.id].append((gpu_id_row, gpu_id_column))
                # sort again in case the max row not be the max
                #if not self.high_affinity:
                #    cur_free_gpus2.sort(key = lambda x: x[0], reverse = True)
                if j > 0:
                    self.hungry_requests[cur_waiting_request.id] = cur_waiting_request
                    requests_cur_steps[cur_waiting_request.id] = 0
                    self.requests_last_steps[cur_waiting_request.id] = 0
                if self.high_affinity:
                    cur_waiting_request.workers_ids = copy.deepcopy(self.requests_workers_ids[cur_waiting_request.id])
                else:
                    cur_waiting_request.workers_ids2 = copy.deepcopy(self.requests_workers_ids2[cur_waiting_request.id])
                self.waiting_requests.popleft()
                return cur_waiting_request
        return None
    
    def static_sp_fcfs_scheduler(self, sp_size: int) -> Request:
        if self.high_affinity:
            cur_free_gpus: Queue[int] = Queue()
            for gpu_id, status in enumerate(self.gpu_status):
                if status == 0:
                    cur_free_gpus.put(gpu_id)
            if cur_free_gpus.qsize() < sp_size:
                return None
        else:
            cur_free_gpus2 = self.get_free_gpus_topology()
            cur_free_gpus2.sort(key = lambda x: len(x), reverse = True)
            if len(cur_free_gpus2[0]) < sp_size:
                return None
        if self.waiting_requests:
            cur_waiting_request = self.waiting_requests[0]
            # help to end the first loop
            if cur_waiting_request == "exit":
                self.waiting_requests.popleft()
                return cur_waiting_request
            
            for _ in range(sp_size):
                if self.high_affinity:
                    gpu_id = cur_free_gpus.get()
                    self.gpu_status[gpu_id] = 1
                    if cur_waiting_request.id not in self.requests_workers_ids:
                        self.requests_workers_ids[cur_waiting_request.id] = [gpu_id]
                    else:
                        self.requests_workers_ids[cur_waiting_request.id].append(gpu_id)
                else:
                    gpu_id_row, gpu_id_column = cur_free_gpus2[0].pop(0) # update the max row itself
                    #cur_free_gpus2[0][0] -= 1 # update free gpu num in the max row
                    self.gpu_status2[gpu_id_row][gpu_id_column] = 1
                    if cur_waiting_request.id not in self.requests_workers_ids2:
                        self.requests_workers_ids2[cur_waiting_request.id] = [(gpu_id_row, gpu_id_column)]
                    else:
                        self.requests_workers_ids2[cur_waiting_request.id].append((gpu_id_row, gpu_id_column))
            # FCFS -> no need to sort cur_free_gpus2
            #if not self.high_affinity:
            #    cur_free_gpus2.sort(key = lambda x: x[0], reverse = True)
            if self.high_affinity:
                cur_waiting_request.workers_ids = copy.deepcopy(self.requests_workers_ids[cur_waiting_request.id])
            else:
                cur_waiting_request.workers_ids2 = copy.deepcopy(self.requests_workers_ids2[cur_waiting_request.id])
            self.waiting_requests.popleft()
            return cur_waiting_request
        return None

def gs(global_scheduler: GlobalScheduler, sp_size: Optional[int] = None) -> None:
    while True:
        if len(finished_requests) == global_scheduler.jobs_num:
            break
        if sp_size:
            request = global_scheduler.static_sp_fcfs_scheduler(sp_size=sp_size)
        else:
            request = global_scheduler.affinity_aware_hungry_first_priority_schedule()
        if request:
            tasks_queue.put(request)
            # if sp_size:
            #     print("request ", request)
            #     time.sleep(1)
    return

class Engine:
    def __init__(self, log_file_path: str, jobs_num: int, high_affinity: bool = True):
        self.dit_times: Dict[str, Dict[int, float]] = {"144p": {1: 3, 2: 3.4, 4: 3.5}, 
                                                       "240p": {1: 8.3, 2: 4.6, 4: 3.7}, 
                                                       "360p": {1: 19.2, 2: 10.4, 4: 6.1}}
        self.vae_times: Dict[str, Dict[int, float]] = {"144p": {1: 0.16, 2: 0.16, 4: 0.16}, 
                                                       "240p": {1: 0.38, 2: 0.38, 4: 0.38}, 
                                                       "360p": {1: 0.87, 2: 0.87, 4: 0.87}}
        self.denoising_steps: int = 30
        self.log_file_path = log_file_path
        self.jobs_num = jobs_num
        self.high_affinity = high_affinity
    
    def generate_dit(self, id: int, resolution: str, 
                     workers_ids: Optional[List[int]] = None,
                     workers_ids2: Optional[List[Tuple[int, int]]] = None) -> None:
        for step_id in range(self.denoising_steps):
            if id in requests_new_workers_ids or id in requests_new_workers_ids2:
                if self.high_affinity:
                    gap_workers_ids = list(set(requests_new_workers_ids[id]) - set(workers_ids))
                else:
                    gap_workers_ids = list(set(requests_new_workers_ids2[id]) - set(workers_ids2))
            else:
                gap_workers_ids = []
            if gap_workers_ids:
                if self.high_affinity:
                    print(f"request {id} resolution {resolution} old gpus {workers_ids} new gpus {requests_new_workers_ids[id]}") # add for log
                    workers_ids = copy.deepcopy(requests_new_workers_ids[id])
                else:
                    print(f"request {id} resolution {resolution} old gpus {workers_ids2} new gpus {requests_new_workers_ids2[id]}") # add for log
                    workers_ids2 = copy.deepcopy(requests_new_workers_ids2[id])
            if self.high_affinity:    
                cur_sleep_time = self.dit_times[resolution][len(workers_ids)] / self.denoising_steps
            else:
                cur_sleep_time = self.dit_times[resolution][len(workers_ids2)] / self.denoising_steps
            requests_cur_steps[id] = step_id + 1
            time.sleep(cur_sleep_time)
        return
    
    def generate_vae(self, id: int, resolution: str,
                     workers_ids: Optional[List[int]], 
                     workers_ids2: Optional[List[Tuple[int, int]]]) -> None:
        if self.high_affinity:  
            cur_sleep_time = self.vae_times[resolution][len(workers_ids)]
        else:
            cur_sleep_time = self.vae_times[resolution][len(workers_ids2)]
        time.sleep(cur_sleep_time)
        end_time = time.time()
        finished_requests.append(id)
        print(f"request {id} resolution {resolution} ends") # add for log
        with open(self.log_file_path, 'a') as file:
            file.write(f"request {id} ends at {end_time}\n")

def task_consumer(engine: Engine, global_scheduler: GlobalScheduler, high_affinity: Optional[bool] = True, 
                  static: Optional[bool] = False) -> None:
    while True:
        # if len(finished_requests) == engine.jobs_num:
        #     break
        task = tasks_queue.get()
        if task == "exit":
            print("thread exit ", threading.get_native_id())
            break
        print(f"request {task.id} resolution {task.resolution} starts") # add for log
        if task.resolution == "144p" or static:
            if high_affinity:
                dit_thread = threading.Thread(target = engine.generate_dit, args = (task.id, task.resolution, task.workers_ids, None))
            else:
                dit_thread = threading.Thread(target = engine.generate_dit, args = (task.id, task.resolution, None, task.workers_ids2))
            dit_thread.start()
            dit_thread.join()
            if high_affinity:
                vae_thread = threading.Thread(target = engine.generate_vae, args = (task.id, task.resolution, task.workers_ids, None))
            else:
                vae_thread = threading.Thread(target = engine.generate_vae, args = (task.id, task.resolution, None, task.workers_ids2))
            vae_thread.start()
            vae_thread.join()
            if not static:
                global_scheduler.update_gpu_status(last = True, request_id = task.id)
            else:
                global_scheduler.update_gpu_status_static(request_id = task.id)
        else:
            if high_affinity:
                dit_thread = threading.Thread(target = engine.generate_dit, args = (task.id, task.resolution, task.workers_ids, None))
            else:
                dit_thread = threading.Thread(target = engine.generate_dit, args = (task.id, task.resolution, None, task.workers_ids2))
            dit_thread.start()
            dit_thread.join()
            global_scheduler.update_gpu_status(last = False, request_id = task.id)
            if high_affinity:
                vae_thread = threading.Thread(target = engine.generate_vae, args = (task.id, task.resolution, [task.workers_ids[0]], 
                                                                                    None))
            else:
                vae_thread = threading.Thread(target = engine.generate_vae, args = (task.id, task.resolution, None, 
                                                                                    [task.workers_ids2[0]]))
            vae_thread.start()
            vae_thread.join()
            global_scheduler.update_gpu_status(last = True, request_id = task.id)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file-path", type = str, default = "/home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/logs_temp/")
    parser.add_argument("--instances-num", type = int, default = 8)
    parser.add_argument("--gpus-per-instance", type = int, default = 8)
    parser.add_argument("--low", type = int, default = 1)
    parser.add_argument("--middle", type = int, default = 1)
    parser.add_argument("--high", type = int, default = 1)
    parser.add_argument("--requests-num", type = int, default = 128)
    parser.add_argument("--batch", type = int, default = 1)
    parser.add_argument("--arrival-ratio", type = float, default = 8.0)
    parser.add_argument("--high-affinity", type = int, default = 1)
    #parser.add_argument("--static", type = bool, default = False)
    parser.add_argument("--sp-size", type = int, default = 4)
    args = parser.parse_args()

    os.makedirs(args.log_file_path, exist_ok=True)

    random.seed(42)

    resolutions = ["144p", "240p", "360p"]
    ratios: List[int] = [args.low, args.middle, args.high]
    total_ratios = sum(ratios)
    total_nums = [round(args.requests_num * (ratio / total_ratios)) for ratio in ratios] 
    add_resolutions: List[str] = []
    for i, num in enumerate(total_nums):
        for _ in range(num):
            add_resolutions.append(resolutions[i])
    random.shuffle(add_resolutions)

    if args.high_affinity:
        high_affinity = True
    else:
        high_affinity = False

    if args.batch:
        add_requests: List[Request] = []
        for i, resolution in enumerate(add_resolutions):
            add_requests.append(Request(id = i, resolution = resolution))
        for j in range(2):
            if j == 0:
                log_file_path = args.log_file_path + "ddit.txt"
            else:
                log_file_path = args.log_file_path + "static.txt"
            engine = Engine(log_file_path = log_file_path, jobs_num = args.requests_num, high_affinity = high_affinity)
            globalscheduler = GlobalScheduler(instances_num = args.instances_num, jobs_num = args.requests_num, 
                                              high_affinity = high_affinity,
                                              gpus_per_instance = args.gpus_per_instance)
            
            if j == 1:
                #reset when iteration 
                finished_requests = []
                requests_new_workers_ids = {}
                requests_new_workers_ids2 = {}
                requests_cur_steps = {}
                tasks_queue = Queue()
            
            consumers_num = args.instances_num * (args.gpus_per_instance // args.sp_size)
            for request in add_requests:
                globalscheduler.add_request(request = request)
            
            #for consumer exit
            print("consumers_num " , consumers_num, len(globalscheduler.waiting_requests))
            for _ in range(consumers_num):
                globalscheduler.add_request(request = "exit")
                
            total_threads: List[threading.Thread] = []
            for _ in range(consumers_num):
                if j == 0:
                    consumer = threading.Thread(target = task_consumer, args = (engine, globalscheduler, high_affinity, False))
                else:
                    consumer = threading.Thread(target = task_consumer, args = (engine, globalscheduler, high_affinity, True))
                consumer.start()
                total_threads.append(consumer)
            if j == 0:
                gs_thread = threading.Thread(target = gs, args = (globalscheduler, None))
            else:
                gs_thread = threading.Thread(target = gs, args = (globalscheduler, args.sp_size))
            with open(log_file_path, 'a') as file:
                file.write(f"start at {time.time()}\n")
            gs_thread.start()
            total_threads.append(gs_thread)
            for thread in total_threads:
                thread.join()
    else:
        for j in range(2):
            if j == 0:
                log_file_path1 = args.log_file_path + "ddit1.txt"
                log_file_path2 = args.log_file_path + "ddit2.txt"
            else:
                log_file_path1 = args.log_file_path + "static1.txt"
                log_file_path2 = args.log_file_path + "static2.txt"
            engine = Engine(log_file_path = log_file_path2, jobs_num = args.requests_num, high_affinity = high_affinity)
            globalscheduler = GlobalScheduler(instances_num = args.instances_num, jobs_num = args.requests_num, 
                                              high_affinity = high_affinity,
                                              gpus_per_instance = args.gpus_per_instance)
            
            if j == 1:
                #reset when iteration 
                finished_requests = []
                requests_new_workers_ids = {}
                requests_new_workers_ids2 = {}
                requests_cur_steps = {}
                tasks_queue = Queue()
            
            consumers_num = args.instances_num * (args.gpus_per_instance // args.sp_size)
            total_threads: List[threading.Thread] = []
            for _ in range(consumers_num):
                if j == 0:
                    consumer = threading.Thread(target = task_consumer, args = (engine, globalscheduler, high_affinity, False))
                else:
                    consumer = threading.Thread(target = task_consumer, args = (engine, globalscheduler, high_affinity, True))
                consumer.start()
                total_threads.append(consumer)
            if j == 0:
                gs_thread = threading.Thread(target = gs, args = (globalscheduler, None))
            else:
                gs_thread = threading.Thread(target = gs, args = (globalscheduler, args.sp_size))
            gs_thread.start()
            total_threads.append(gs_thread)
            for i in range(args.requests_num):
                request = Request(id = i, resolution = add_resolutions[i])
                globalscheduler.add_request(request = request)
                start_time = time.time()
                with open(log_file_path1, 'a') as file:
                    file.write(f"request {i} starts at {start_time}\n")
                time.sleep(1 / args.arrival_ratio)
            
            #for consumer exit
            for _ in range(consumers_num):
                globalscheduler.add_request(request = "exit")
            
            for thread in total_threads:
                thread.join()