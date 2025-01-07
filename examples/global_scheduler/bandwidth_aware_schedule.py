from typing import Dict, List, Tuple, Deque
import threading
import copy
import time
import argparse
import random

class Request:
    def __init__(self, id: int, resolution: str) -> None:
        self.id = id
        self.resolution = resolution

class Resources:
    def __init__(self, instances_num: int, gpus_per_instance: int, log_path: str) -> None:
        self.free_gpus_list = [[0 for _ in range(gpus_per_instance)] for _ in range(instances_num)]
        self.free_gpus_lock = threading.Lock()
        self.file_lock = threading.Lock()
        self.new_gpus = threading.Event()
        self.new_gpus_lock = threading.Lock()
        self.hungry_requests: Dict[Request, threading.Event] = {}
        self.hungry_requests_lock = threading.Lock()
        self.opt_gpu_nums: Dict[str, int] = {"144p": 1, "240p": 2, "360p": 4}
        self.dit_times: Dict[str, Dict[int, float]] = {"144p": {1: 3, 2: 3.4, 4: 3.5}, "240p": {1: 8.3, 2: 4.6, 4: 3.7}, 
                                                       "360p": {1: 19.2, 2: 10.4, 4: 6.1}}
        self.vae_times: Dict[str, Dict[int, float]] = {"144p": {1: 0.16, 2: 0.16, 4: 0.16}, "240p": {1: 0.38, 2: 0.38, 4: 0.38}, 
                                                       "360p": {1: 0.87, 2: 0.87, 4: 0.87}}
        self.last_step: Dict[int, int] = {}
        self.cur_step: Dict[int, int] = {}
        self.cur_starv_time: Dict[int, float] = {}
        self.cur_allocated_gpus: Dict[int, List[Tuple[int, int]]] = {}
        self.denoise_steps = 30
        self.log_path = log_path
        self.waiting_requests: Deque[Request] = Deque()
    
    def write_logs(self, log_time: float, id: int) -> None:
        with self.file_lock:
            with open(self.log_path, 'a') as file:
                if id >= 0:
                    file.write(f"request {id} ends at {log_time}")
                else:
                    file.write(f"requests start at {log_time}")

    def add_request(self, request: Request) -> None:
        self.waiting_requests.append(request)

    def try_best_allocate(self, request: Request, allocated_gpu_num: int, 
                          allocated_gpu_list: List[Tuple[int, int]]) -> Tuple[bool, List[Tuple[int, int]], float, float]:
        cur_opt_gpu_num = self.opt_gpu_nums[request.resolution]
        wanted_gpu_num_list: List[int] = []
        while cur_opt_gpu_num > 0:
            if cur_opt_gpu_num - allocated_gpu_num > 0:
                wanted_gpu_num_list.append(cur_opt_gpu_num - allocated_gpu_num)
            cur_opt_gpu_num //= 2
        with self.free_gpus_lock:
            max_allocated_gpu_list: List[Tuple[int, int]] = []
            for instance_id in len(self.free_gpus_list):
                cur_allocated_gpu_list: List[Tuple[int, int]] = []
                for gpu_id in len(self.free_gpus_list[instance_id]):
                    if self.free_gpus_list[instance_id][gpu_id] == 0:
                        cur_allocated_gpu_list.append((instance_id, gpu_id))
                if len(cur_allocated_gpu_list) > len(max_allocated_gpu_list):
                    max_allocated_gpu_list = copy.deepcopy(cur_allocated_gpu_list)
                    if len(max_allocated_gpu_list) >= wanted_gpu_num_list[0]:
                        for k, l in max_allocated_gpu_list[0: wanted_gpu_num_list[0]]:
                            self.free_gpus_list[k][l] = 1
                            allocated_gpu_list.append((k, l))
                        if request in self.hungry_requests:
                            with self.hungry_requests_lock:
                                if not self.hungry_requests[request].is_set():
                                    self.hungry_requests[request].set()
                                del self.hungry_requests[request]
                                del self.last_step[request.id]
                                del self.cur_step[request.id]
                                del self.cur_starv_time[request.id]
                                #self.cur_allocated_gpus[request.id] = allocated_gpu_list
                        return (True, allocated_gpu_list, self.dit_times[request.resolution][len(allocated_gpu_list)], 
                                        self.vae_times[request.resolution][len(allocated_gpu_list)])
            for demand_gpu_num in wanted_gpu_num_list[1: -1]:
                if len(max_allocated_gpu_list) >= demand_gpu_num:
                    for i, j in max_allocated_gpu_list[0: demand_gpu_num]:
                        self.free_gpus_list[i][j] = 1
                        allocated_gpu_list.append((i, j))
                    if request not in self.hungry_requests:
                        with self.hungry_requests_lock:
                            self.hungry_requests[request] = threading.Event()
                            self.last_step[request.id] = 0
                            self.cur_step[request.id] = 0
                            self.cur_starv_time[request.id] = (self.dit_times[request.resolution][len(allocated_gpu_list)] 
                                                            - self.dit_times[request.resolution][self.opt_gpu_nums[request.resolution]]) / self.denoise_steps
                            self.cur_allocated_gpus[request.id] = allocated_gpu_list
                        return (True, allocated_gpu_list, self.dit_times[request.resolution][len(allocated_gpu_list)], 
                                self.vae_times[request.resolution][len(allocated_gpu_list)])
                    else:
                        with self.hungry_requests_lock:
                            if not self.hungry_requests[request].is_set():
                                self.hungry_requests[request].set()
                            self.last_step[request.id] = self.cur_step[request.id]
                            self.cur_starv_time[request.id] = (self.dit_times[request.resolution][len(allocated_gpu_list)]
                                                            - self.dit_times[request.resolution][self.opt_gpu_nums[request.resolution]]) / self.denoise_steps
                            #self.cur_allocated_gpus[request.id] = allocated_gpu_list
                        return (True, allocated_gpu_list, self.dit_times[request.resolution][len(allocated_gpu_list)], 
                                self.vae_times[request.resolution][len(allocated_gpu_list)])
            return (False, None, None, None)
    
    def release_resources(self, allocated_gpu_list: List[Tuple[int, int]]) -> None:
        with self.free_gpus_lock:
            for i, j in allocated_gpu_list:
                self.free_gpus_list[i][j] = 0
            with self.new_gpus_lock:
                if not self.new_gpus.is_set():
                    self.new_gpus.set()

def global_schedule(resource_pool: Resources) -> None:
    while resource_pool.hungry_requests or resource_pool.waiting_requests:
        if resource_pool.new_gpus.is_set():
            if resource_pool.hungry_requests:
                requests = list(resource_pool.hungry_requests.keys())
                requests.sort(key = lambda x: resource_pool.cur_starv_time[x.id] * (resource_pool.cur_step[x.id] 
                                                                                    - resource_pool.last_step[x.id]), reverse = True)
                for request in requests:
                    _, _, _, _ = resource_pool.try_best_allocate(request = request, 
                                                                    allocated_gpu_num = len(resource_pool.cur_allocated_gpus[request.id]),
                                                                    allocated_gpu_list = resource_pool.cur_allocated_gpus[request.id])
            with resource_pool.new_gpus_lock:
                resource_pool.new_gpus.clear()

def thread_function(request: Request, resource_pool: Resources, allocated_gpu_list: List[Tuple[int, int]],
                    dit_step_time: float, vae_time: float) -> None:
    print(f"Request {request.id} Starts")
    cur_step = 0
    while cur_step < resource_pool.denoise_steps:
        if resource_pool.hungry_requests[request.id] is not None and resource_pool.hungry_requests[request.id].is_set():
            allocated_gpu_list = resource_pool.cur_allocated_gpus[request.id]
            dit_step_time = resource_pool.dit_times[request.resolution][len(allocated_gpu_list)] / resource_pool.denoise_steps
            cur_step += 1
            with resource_pool.hungry_requests_lock:
                resource_pool.cur_step[request.id] = cur_step
                resource_pool.hungry_requests[request.id].clear()
            time.sleep(dit_step_time)
        else:
            time.sleep(dit_step_time)
            cur_step += 1
    resource_pool.release_resources(allocated_gpu_list = allocated_gpu_list[0: -2])
    time.sleep(vae_time)
    resource_pool.release_resources(allocated_gpu_list = allocated_gpu_list[-1])
    end_time = time.time()
    print(f"Request {request.id} Ends")
    resource_pool.write_logs(log_time = end_time, id = request.id)

def ddit_schedule(resource_pool: Resources) -> None:
    activate_threads: List[threading.Thread] = []
    global_scheduler = threading.Thread(target = global_schedule ,args = (resource_pool))
    global_scheduler.start()
    activate_threads.append(global_scheduler)
    resource_pool.write_logs(log_time = time.time(), id = -1)
    print(f"Test Starts!")
    while resource_pool.waiting_requests:
        cur_request = resource_pool.waiting_requests.popleft()
        can_start, allocated_gpu_list, dit_time, vae_time = resource_pool.try_best_allocate(request = cur_request,
                                                                                                          allocated_gpu_num = 0,
                                                                                                          allocated_gpu_list = [])
        if can_start:
            cur_thread = threading.Thread(target = thread_function, args = (cur_request, resource_pool, allocated_gpu_list, 
                                                                            dit_time / resource_pool.denoise_steps, vae_time))
            cur_thread.start()
            activate_threads.append(cur_thread)
        else:
            resource_pool.waiting_requests.append(cur_request)
    for cur_thread in activate_threads:
        cur_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type = str, default = "/home/jovyan/hhy/VideoSys/examples/global_scheduler/logs_temp.txt")
    parser.add_argument("--instances", type = int, default = 8)
    parser.add_argument("--gpus", type = int, default = 8)
    parser.add_argument("--weight1", type = int, default = 1)
    parser.add_argument("--weight2", type = int, default = 1)
    parser.add_argument("--weight3", type = int, default = 1)
    parser.add_argument("--num", type = int, default = 128)
    args = parser.parse_args()
    random.seed(42)
    resolutions = ["144p", "240p", "360p"]
    requests_resolutions: List[str] = []
    total_weight = args.weight1 + args.weight2 + args.weight3
    for _ in range(round((args.weight1 / total_weight) * args.num)):
        requests_resolutions.append(resolutions[0])
    for _ in range(round((args.weight2 / total_weight) * args.num)):
         requests_resolutions.append(resolutions[1])
    for _ in range(round((args.weight3 / total_weight) * args.num)):
         requests_resolutions.append(resolutions[2])
    random.shuffle(requests_resolutions)
    resource_pool = Resources(instances_num = args.instances, gpus_per_instance = args.gpus, log_path = args.log)
    for i, resolution in enumerate(requests_resolutions):
        resource_pool.add_request(request = Request(id = i, resolution = resolution))
    ddit_schedule(resource_pool = resource_pool)