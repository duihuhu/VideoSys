from typing import Dict, List, Tuple, Deque, Optional
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
    def __init__(self, instances_num: int, gpus_per_instance: int, log_path: str, per_group_num: Optional[int]) -> None:
        self.free_gpus_list = [[0 for _ in range(gpus_per_instance)] for _ in range(instances_num)]
        self.free_gpus_lock = threading.Lock()
        self.unify_free_gpus_lock = threading.Lock()
        self.free_gpus_num = instances_num * gpus_per_instance
        self.file_lock = threading.Lock()
        self.new_gpus = threading.Event()
        self.new_gpus_lock = threading.Lock()
        self.hungry_requests: Dict[int, threading.Event] = {}
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
        self.unify_network_cur_allocated_gpus: Dict[int, int] = {}
        self.denoise_steps = 30
        self.log_path = log_path
        self.waiting_requests: Deque[Request] = Deque()
        self.helper_requests: List[Request] = []
        self.groups = instances_num * gpus_per_instance // per_group_num
        self.groups_lock = threading.Lock()
        self.per_group_num = per_group_num
        self.end_times: Dict[int, float] = {}
    
    def write_logs(self, log_time: float, id: int) -> None:
        with self.file_lock:
            with open(self.log_path, 'a') as file:
                if id >= 0:
                    file.write(f"request {id} ends at {log_time}\n")
                else:
                    file.write(f"requests start at {log_time}\n")

    def add_request(self, request: Request) -> None:
        self.waiting_requests.append(request)
        self.helper_requests.append(request)
    
    def allocate_resources(self, request: Request) -> Tuple[bool, float, float]:
        with self.groups_lock:
            if self.groups >= 1:
                self.groups -= 1
                return (True, self.dit_times[request.resolution][self.per_group_num], self.vae_times[request.resolution][self.per_group_num])
            else:
                return (False, None, None)
    
    def unify_network_allocate(self, request: Request, demand_gpu_num) -> Tuple[bool, int, float, float]:
        cur_opt_gpu_num = self.opt_gpu_nums[request.resolution]
        cur_demand_gpu_nums: List[int] = []
        while cur_opt_gpu_num > 0:
            if cur_opt_gpu_num - demand_gpu_num > 0:
                cur_demand_gpu_nums.append(cur_opt_gpu_num - demand_gpu_num)
            cur_opt_gpu_num //= 2
        with self.unify_free_gpus_lock:
            if self.free_gpus_num >= cur_demand_gpu_nums[0]:
                self.free_gpus_num -= cur_demand_gpu_nums[0]
                cur_allocated_gpus_num = cur_demand_gpu_nums[0] + demand_gpu_num    
                if request.id in self.hungry_requests:
                    #with self.hungry_requests_lock:
                        #if not self.hungry_requests[request.id].is_set():
                    self.hungry_requests[request.id].set()
                    del self.hungry_requests[request.id]
                    del self.last_step[request.id]
                    del self.cur_step[request.id]
                    del self.cur_starv_time[request.id]
                    self.unify_network_cur_allocated_gpus[request.id] = cur_allocated_gpus_num
                return (True, cur_allocated_gpus_num, self.dit_times[request.resolution][cur_allocated_gpus_num], 
                self.vae_times[request.resolution][cur_allocated_gpus_num])
            for cur_demand_gpu_num in cur_demand_gpu_nums[1: ]:
                if self.free_gpus_num >= cur_demand_gpu_num:
                    self.free_gpus_num -= cur_demand_gpu_num
                    cur_allocated_gpus_num = cur_demand_gpu_num + demand_gpu_num
                    if request.id not in self.hungry_requests:
                        #with self.hungry_requests_lock:
                        self.hungry_requests[request.id] = threading.Event()
                        self.last_step[request.id] = 0
                        self.cur_step[request.id] = 0
                        self.cur_starv_time[request.id] = (self.dit_times[request.resolution][cur_allocated_gpus_num] 
                        - self.dit_times[request.resolution][self.opt_gpu_nums[request.resolution]]) / self.denoise_steps
                        self.unify_network_cur_allocated_gpus[request.id] = cur_allocated_gpus_num                              
                        return (True, cur_allocated_gpus_num, self.dit_times[request.resolution][cur_allocated_gpus_num], 
                                self.vae_times[request.resolution][cur_allocated_gpus_num])
                    else:
                        #with self.hungry_requests_lock:
                            #if not self.hungry_requests[request.id].is_set():
                        self.hungry_requests[request.id].set()
                        self.last_step[request.id] = self.cur_step[request.id]
                        self.cur_starv_time[request.id] = (self.dit_times[request.resolution][cur_allocated_gpus_num]
                        - self.dit_times[request.resolution][self.opt_gpu_nums[request.resolution]]) / self.denoise_steps
                        self.unify_network_cur_allocated_gpus[request.id] = cur_allocated_gpus_num
                        return (True, cur_allocated_gpus_num, self.dit_times[request.resolution][cur_allocated_gpus_num], 
                                self.vae_times[request.resolution][cur_allocated_gpus_num])
            return (False, None, None, None)

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
            for instance_id in range(len(self.free_gpus_list)):
                cur_allocated_gpu_list: List[Tuple[int, int]] = []
                for gpu_id in range(len(self.free_gpus_list[instance_id])):
                    if self.free_gpus_list[instance_id][gpu_id] == 0:
                        cur_allocated_gpu_list.append((instance_id, gpu_id))
                if len(cur_allocated_gpu_list) > len(max_allocated_gpu_list):
                    max_allocated_gpu_list = copy.deepcopy(cur_allocated_gpu_list)
                    if len(max_allocated_gpu_list) >= wanted_gpu_num_list[0]:
                        for k, l in max_allocated_gpu_list[0: wanted_gpu_num_list[0]]:
                            self.free_gpus_list[k][l] = 1
                            allocated_gpu_list.append((k, l))
                        if request.id in self.hungry_requests:
                            #with self.hungry_requests_lock:
                                #if not self.hungry_requests[request.id].is_set():
                            self.hungry_requests[request.id].set()
                            del self.hungry_requests[request.id]
                            del self.last_step[request.id]
                            #del self.cur_step[request.id]
                            del self.cur_starv_time[request.id]
                            self.cur_allocated_gpus[request.id] = allocated_gpu_list
                        return (True, allocated_gpu_list, self.dit_times[request.resolution][len(allocated_gpu_list)], 
                        self.vae_times[request.resolution][len(allocated_gpu_list)])
            for demand_gpu_num in wanted_gpu_num_list[1: ]:
                if len(max_allocated_gpu_list) >= demand_gpu_num:
                    for i, j in max_allocated_gpu_list[0: demand_gpu_num]:
                        self.free_gpus_list[i][j] = 1
                        allocated_gpu_list.append((i, j))
                    if request.id not in self.hungry_requests:
                        #with self.hungry_requests_lock:
                        self.hungry_requests[request.id] = threading.Event()
                        self.last_step[request.id] = 0
                        self.cur_step[request.id] = 0
                        self.cur_starv_time[request.id] = (self.dit_times[request.resolution][len(allocated_gpu_list)] 
                        - self.dit_times[request.resolution][self.opt_gpu_nums[request.resolution]]) / self.denoise_steps
                        self.cur_allocated_gpus[request.id] = allocated_gpu_list
                        return (True, allocated_gpu_list, self.dit_times[request.resolution][len(allocated_gpu_list)], 
                                self.vae_times[request.resolution][len(allocated_gpu_list)])
                    else:
                        #with self.hungry_requests_lock:
                            #if not self.hungry_requests[request.id].is_set():
                        self.hungry_requests[request.id].set()
                        self.last_step[request.id] = self.cur_step[request.id]
                        self.cur_starv_time[request.id] = (self.dit_times[request.resolution][len(allocated_gpu_list)]
                        - self.dit_times[request.resolution][self.opt_gpu_nums[request.resolution]]) / self.denoise_steps
                        self.cur_allocated_gpus[request.id] = allocated_gpu_list
                        return (True, allocated_gpu_list, self.dit_times[request.resolution][len(allocated_gpu_list)], 
                                self.vae_times[request.resolution][len(allocated_gpu_list)])
            return (False, None, None, None)
    
    def release_resources(self, allocated_gpu_list: List[Tuple[int, int]], last: bool) -> None:
        with self.free_gpus_lock:
            if last:
                i, j = allocated_gpu_list[-1]
                self.free_gpus_list[i][j] = 0
            else:
                for i in range(0, len(allocated_gpu_list) - 1):
                    m, n = allocated_gpu_list[i]
                    self.free_gpus_list[m][n] = 0
            #with self.new_gpus_lock:
                #if not self.new_gpus.is_set():
            self.new_gpus.set()
    
    def unify_network_release(self, allocated_gpu_num: int) -> None:
        with self.unify_free_gpus_lock:
            self.free_gpus_num += allocated_gpu_num
            #with self.new_gpus_lock:
                #if not self.new_gpus.is_set():
            self.new_gpus.set()
    
    def group_release_resources(self) -> None:
        with self.groups_lock:
            self.groups += 1

def global_schedule(resource_pool: Resources, unify: Optional[bool] = False) -> None:
    while resource_pool.hungry_requests or resource_pool.waiting_requests:
        if resource_pool.new_gpus.is_set():
            if resource_pool.hungry_requests:
                requests_ids = list(resource_pool.hungry_requests.keys())
                requests_ids.sort(key = lambda x: resource_pool.cur_starv_time[x] * (resource_pool.cur_step[x] 
                                                                                    - resource_pool.last_step[x]), reverse = True)
                for id in requests_ids:
                    if unify:
                        _, _, _, _ = resource_pool.unify_network_allocate(request = resource_pool.helper_requests[id], 
                        demand_gpu_num = resource_pool.unify_network_cur_allocated_gpus[id])
                    else:
                        _, _, _, _ = resource_pool.try_best_allocate(request = resource_pool.helper_requests[id], 
                                                                     allocated_gpu_num = len(resource_pool.cur_allocated_gpus[id]),
                                                                     allocated_gpu_list = resource_pool.cur_allocated_gpus[id])
            #with resource_pool.new_gpus_lock:
            resource_pool.new_gpus.clear()

def thread_function(request: Request, resource_pool: Resources, allocated_gpu_list: List[Tuple[int, int]],
                    dit_step_time: float, vae_time: float, allocated_gpu_num: Optional[int], unify: Optional[bool] = False) -> None:
    print(f"Request {request.id} Starts")
    cur_step = 0
    while cur_step < resource_pool.denoise_steps:
        if request.id in resource_pool.hungry_requests and resource_pool.hungry_requests[request.id].is_set():
            if unify:
                allocated_gpu_num = resource_pool.unify_network_cur_allocated_gpus[request.id]
                dit_step_time = resource_pool.dit_times[request.resolution][allocated_gpu_num] / resource_pool.denoise_steps
            else:
                allocated_gpu_list = resource_pool.cur_allocated_gpus[request.id]
                dit_step_time = resource_pool.dit_times[request.resolution][len(allocated_gpu_list)] / resource_pool.denoise_steps
            cur_step += 1
            #with resource_pool.hungry_requests_lock:
            resource_pool.cur_step[request.id] = cur_step
            resource_pool.hungry_requests[request.id].clear()
            time.sleep(dit_step_time)
        else:
            time.sleep(dit_step_time)
            cur_step += 1
    
    if len(allocated_gpu_list) >= 2 or allocated_gpu_num >= 2:
        if unify:
            resource_pool.unify_network_release(allocated_gpu_num - 1)
        else:
            resource_pool.release_resources(allocated_gpu_list = allocated_gpu_list, last = False)
    time.sleep(vae_time)
    if unify:
        resource_pool.unify_network_release(1)
    else:
        resource_pool.release_resources(allocated_gpu_list = allocated_gpu_list, last = True)
    end_time = time.time()
    print(f"Request {request.id} Ends")
    resource_pool.end_times[request.id] = end_time
    if request.id in resource_pool.hungry_requests:
        del resource_pool.hungry_requests[request.id]
    #resource_pool.write_logs(log_time = end_time, id = request.id)

def group_thread_function(request: Request, resource_pool: Resources, dit_time: float, vae_time: float) -> None:
    print(f"Request {request.id} Starts")
    for _ in range(resource_pool.denoise_steps):
        time.sleep(dit_time / resource_pool.denoise_steps)
    time.sleep(vae_time)
    resource_pool.group_release_resources()
    end_time = time.time()
    print(f"Request {request.id} Ends")
    resource_pool.end_times[request.id] = end_time
    #resource_pool.write_logs(log_time = end_time, id = request.id)

def ddit_schedule(resource_pool: Resources, group: Optional[bool] = False, unify: Optional[bool] = False,
                  policy: Optional[str] = "Non-Policy") -> None:
    activate_threads: List[threading.Thread] = []
    if group:
        start_time = time.time()
        #resource_pool.write_logs(log_time = time.time(), id = -1)
        print(f"Test Starts!")
        while resource_pool.waiting_requests:
            cur_request = resource_pool.waiting_requests.popleft()
            can_start, dit_time, vae_time = resource_pool.allocate_resources(request = cur_request)
            if can_start:
                cur_thread = threading.Thread(target = group_thread_function, args = (cur_request, resource_pool, dit_time, vae_time))
                cur_thread.start()
                activate_threads.append(cur_thread)
            else:
                resource_pool.waiting_requests.append(cur_request)
    else:
        global_scheduler = threading.Thread(target = global_schedule, args = (resource_pool, unify), name = "global_scheduler")
        global_scheduler.start()
        activate_threads.append(global_scheduler)
        start_time = time.time()    
        #resource_pool.write_logs(log_time = time.time(), id = -1)
        print(f"Test Starts!")
        while resource_pool.waiting_requests:
            cur_request = resource_pool.waiting_requests.popleft()
            allocated_gpu_num = 0
            allocated_gpu_list: List[Tuple[int, int]] = []
            if unify:
                can_start, allocated_gpu_num, dit_time, vae_time = resource_pool.unify_network_allocate(request = cur_request, 
                demand_gpu_num = 0)
            else:
                can_start, allocated_gpu_list, dit_time, vae_time = resource_pool.try_best_allocate(request = cur_request, 
                allocated_gpu_num = 0, 
                allocated_gpu_list = [])
            if can_start:
                cur_thread = threading.Thread(target = thread_function, args = (cur_request, resource_pool, allocated_gpu_list, 
                                                                                dit_time / resource_pool.denoise_steps, vae_time,
                                                                                allocated_gpu_num, unify), name = f"request_{cur_request.id}")
                cur_thread.start()
                activate_threads.append(cur_thread)
            else:
                resource_pool.waiting_requests.append(cur_request)
    for cur_thread in activate_threads:
        cur_thread.join()
    durations = []
    for _, duration in resource_pool.end_times.items():
        #print(f"ID: {id}, Duration: {duration}")
        durations.append(duration - start_time)
    with open(resource_pool.log_path, 'a') as file:
        file.write(f"Total {len(durations)} Requests, {policy} Average Duration: {sum(durations) / len(durations)}\n")
    #for cur_thread in activate_threads:
    #    cur_thread.join()
        #print(threading.enumerate())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type = str, default = "/home/jovyan/hhy/VideoSys/examples/global_scheduler/logs_temp.txt")
    parser.add_argument("--instances", type = int, default = 8)
    parser.add_argument("--gpus", type = int, default = 8)
    parser.add_argument("--weight1", type = int, default = 1)
    parser.add_argument("--weight2", type = int, default = 1)
    parser.add_argument("--weight3", type = int, default = 8)
    parser.add_argument("--num", type = int, default = 128)
    parser.add_argument("--gnum", type = int, default = 4)
    parser.add_argument("--group", action = 'store_true', default = False)
    parser.add_argument("--unify", action = 'store_true', default = False)
    args = parser.parse_args()
    print(args)
    
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
    
    policies = ["Bandwidth", "Unify", "Group"]
    for policy in policies:
        resource_pool = Resources(instances_num = args.instances, gpus_per_instance = args.gpus, log_path = args.log, per_group_num = args.gnum)
        for i, resolution in enumerate(requests_resolutions):
            resource_pool.add_request(request = Request(id = i, resolution = resolution))
        if policy == "Bandwidth":
            ddit_schedule(resource_pool = resource_pool, group = False, unify = False, policy = policy)
        elif policy == "Unify":
            ddit_schedule(resource_pool = resource_pool, group = False, unify = True, policy = policy)
        else:
            ddit_schedule(resource_pool = resource_pool, group = True, unify = False, policy = policy)