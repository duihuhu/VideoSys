from typing import Dict, Deque, List
from queue import Queue

class Request:
    def __init__(self, id: int, resolution: str):
        self.id = id
        self.resolution = resolution
 
class GlobalScheduler:
    def __init__(self, instances_num: int):
        self.gpu_status = [0 for _ in range(instances_num)]
        self.hungry_requests: Dict[int, Request] = {}
        self.waiting_requests: Deque[Request] = Deque()
        self.requests_workers_ids: Dict[int, List[int]] = {}
        self.requests_cur_steps: Dict[int, int] = {}
        self.requests_last_steps: Dict[int, int] = {}
        self.denoising_steps: int = 30
        self.finished_requests: List[Request] = []
    
    def update_gpu_status(self, last: bool, request_id: int) -> None:
        if last:
            self.gpu_status[self.requests_workers_ids[request_id][0]] = 0
            self.requests_workers_ids.pop(request_id, None)
        else:
            for i in range(1, len(self.requests_workers_ids[request_id])):
                self.gpu_status[self.requests_workers_ids[request_id][i]] = 0
            self.hungry_requests.pop(request_id, None)
            self.requests_cur_steps.pop(request_id, None)
            self.requests_last_steps.pop(request_id, None)
        
    def high_affinity_hungry_first_priority_schedule(self) -> Request:
        cur_free_gpus: Queue[int] = Queue()
        for gpu_id, status in enumerate(self.gpu_status):
            if status == 0:
                cur_free_gpus.put(gpu_id)
        if cur_free_gpus.qsize() < 1:
            return None

        #----------process hungry queue in starvation descending order while num = N#
        temp_hungry_requests = list(self.hungry_requests.values())
        # sort in descending order by starvation time
        temp_hungry_requests.sort(key = lambda x: (self.requests_cur_steps[x.request_id] - self.requests_last_steps[x.request_id])
                                    * (self.dit_times[x.resolution][len(self.requests_workers_ids[x.request_id])] - self.dit_times[x.resolution][self.opt_gps_num[x.resolution]])
                                    / self.denoising_steps
                                    , reverse = True)
        
        update_requests: List[str] = []
        for cur_hungry_request in temp_hungry_requests:
            if cur_free_gpus.qsize() < 1:
                break
            
            cur_wanted_gpus_num = []
            cur_opt_gpus_num = self.opt_gps_num[cur_hungry_request.resolution]
            while cur_opt_gpus_num > 0:
                if cur_opt_gpus_num - len(self.requests_workers_ids[cur_hungry_request.request_id]) > 0:
                    cur_wanted_gpus_num.append(cur_opt_gpus_num - len(self.requests_workers_ids[cur_hungry_request.request_id]))
                cur_opt_gpus_num //= 2

            for i, wanted_gpus_num in enumerate(cur_wanted_gpus_num):
                if cur_free_gpus.qsize() < wanted_gpus_num:
                    continue
                
                for _ in range(wanted_gpus_num):
                    gpu_id = cur_free_gpus.get()
                    self.gpu_status[gpu_id] = 1
                    self.requests_workers_ids[cur_hungry_request.request_id].append(gpu_id)
                
                update_requests.append(cur_hungry_request.request_id)
                if i == 0:
                    self.hungry_requests.pop(cur_hungry_request.request_id, None)
                    self.requests_cur_steps.pop(cur_hungry_request.request_id, None)
                    self.requests_last_steps.pop(cur_hungry_request.request_id, None)
                else:
                    self.requests_last_steps[cur_hungry_request.request_id] = self.requests_cur_steps[cur_hungry_request.request_id]
            
        for request_id in update_requests:
            print(f"request {request_id} new workers ids {self.requests_workers_ids[request_id]}")
            self.update_tasks.put((request_id, self.requests_workers_ids[request_id]))
            
        #----------process waiting queue in FCFS while num = 1----------#
        if self.waiting:
            if cur_free_gpus.qsize() < 1:
                return None
            
            cur_waiting_request = self.waiting[0]

            cur_demand_gpus_num = []
            cur_max_gpus_num = self.opt_gps_num[cur_waiting_request.resolution]
            while cur_max_gpus_num > 0:
                cur_demand_gpus_num.append(cur_max_gpus_num)
                cur_max_gpus_num //= 2

            for j, demand_gpus_num in enumerate(cur_demand_gpus_num):
                if cur_free_gpus.qsize() < demand_gpus_num:
                    continue

                for _ in range(demand_gpus_num):
                    gpu_id = cur_free_gpus.get()
                    self.gpu_status[gpu_id] = 1
                    if cur_waiting_request.request_id not in self.requests_workers_ids:
                        self.requests_workers_ids[cur_waiting_request.request_id] = [gpu_id]
                    else:
                        self.requests_workers_ids[cur_waiting_request.request_id].append(gpu_id)
                
                if j > 0:
                    self.hungry_requests[cur_waiting_request.request_id] = cur_waiting_request
                    self.requests_cur_steps[cur_waiting_request.request_id] = 0
                    self.requests_last_steps[cur_waiting_request.request_id] = 0
                
                self.waiting.popleft()
                return cur_waiting_request
        
        return None