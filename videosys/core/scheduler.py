from typing import Deque, Dict, Tuple, List
from collections import deque
from videosys.core.sequence import SequenceGroup
import requests
import copy
from queue import Queue
import threading
class VideoScheduler:
    def __init__(
        self,
        instances_num: int
    ) -> None:
        # Sequence groups in the WAITING state.
        self.waiting: Deque[SequenceGroup] = deque()
        self.num_gpus = 2
        self.num = 0
        
        self.gpu_status = [0 for _ in range(instances_num)]
        #self.gpu_status_lock = asyncio.Lock()
        
        self.hungry_requests: Dict[str, SequenceGroup] = {}
        self.requests_workers_ids: Dict[str, List[int]] = {}
        self.requests_last_steps: Dict[str, int] = {}
        self.requests_cur_steps: Dict[str, int] = {}
        
        self.dit_times: Dict[str, Dict[int, float]] = {"144p": {1: 3, 2: 3.4, 4: 3.5}, 
                                                       "240p": {1: 8.3, 2: 4.6, 4: 3.7}, 
                                                       "360p": {1: 19.2, 2: 10.4, 4: 6.1}}
        self.opt_gpus_num: Dict[str, int] = {"144p": 1, "240p": 2, "360p": 4}
        self.denoising_steps = 30

        self.update_tasks: Queue[Tuple[str, List[int]]] = Queue()
        self.async_server_url = "http://127.0.0.1:8000/request_workers"
    
    def post_http_request(self, pload, api_url) -> requests.Response:
        headers = {"User-Agent": "Test Client"}
        response = requests.post(api_url, headers = headers, json = pload)
        return response

    def update_requests_cur_steps(self, request_id: str, cur_step: int) -> None:
        if request_id in self.requests_cur_steps:
            self.requests_cur_steps[request_id] = cur_step
    
    def update_dit_workers_ids_for_request(self) -> None:
        while True:
            if self.update_tasks.empty():
                continue
            group_id, worker_ids = self.update_tasks.get()
            pload = {
                "request_id": group_id,
                "worker_ids": worker_ids,
            }
            _ = self.post_http_request(pload, self.async_server_url)
    
    def create_update_threads(self) -> None:
        cur_thread = threading.Thread(target = self.update_dit_workers_ids_for_request)
        cur_thread.daemon = True # kill gs -> kill AsyncSched -> Kill VideoSched -> Kill VideoScheduler -> Kill thread
        cur_thread.start()

    def update_gpu_status(self, last: bool, group_id: str) -> None:
        print(f"before release {self.gpu_status}")
        if last:
            self.gpu_status[self.requests_workers_ids[group_id][0]] = 0
            print(f"request {group_id} release its last worker {self.requests_workers_ids[group_id][0]}")
            self.requests_workers_ids.pop(group_id, None)
            #self.hungry_requests.pop(group_id, None) # this may be useless
        else:
            for i in range(1, len(self.requests_workers_ids[group_id])):
                self.gpu_status[self.requests_workers_ids[group_id][i]] = 0
            print(f"request {group_id} release {self.requests_workers_ids[group_id][1: ]}")
            self.hungry_requests.pop(group_id, None) 
            self.requests_cur_steps.pop(group_id, None)
            self.requests_last_steps.pop(group_id, None)
        print(f"after release {self.gpu_status}")
    
    def hungry_first_priority_schedule(self) -> SequenceGroup:
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
                                    * (self.dit_times[x.resolution][len(self.requests_workers_ids[x.request_id])] - self.dit_times[x.resolution][self.opt_gpus_num[x.resolution]])
                                    / self.denoising_steps
                                    , reverse = True)
        
        update_requests: List[str] = []
        for cur_hungry_request in temp_hungry_requests:
            if cur_free_gpus.qsize() < 1:
                break
            
            cur_wanted_gpus_num = []
            cur_opt_gpus_num = self.opt_gpus_num[cur_hungry_request.resolution]
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
                    self.requests_cur_steps.pop(cur_hungry_request.request_id, None)
                    self.requests_last_steps.pop(cur_hungry_request.request_id, None)
                    self.hungry_requests.pop(cur_hungry_request.request_id, None)
                else:
                    if cur_hungry_request.request_id in self.requests_last_steps:
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
            cur_max_gpus_num = self.opt_gpus_num[cur_waiting_request.resolution]
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
                    self.requests_cur_steps[cur_waiting_request.request_id] = 0
                    self.requests_last_steps[cur_waiting_request.request_id] = 0
                    self.hungry_requests[cur_waiting_request.request_id] = cur_waiting_request
                
                cur_waiting_request.worker_ids = copy.deepcopy(self.requests_workers_ids[cur_waiting_request.request_id])
                self.waiting.popleft()
                return cur_waiting_request
        
        return None
                
    #async def update_and_schedule
    def update_and_schedule(self, last: bool, group_id: str) -> None:   
        #cur_free_gpus_list = []
        #async with self.gpu_status_lock:
        print(f"before release {self.gpu_status}")
        if last:
            self.gpu_status[self.requests_workers_ids[group_id][0]] = 0
            print(f"request {group_id} release the last worker {self.requests_workers_ids[group_id][0]}")
            
            self.hungry_requests.pop(group_id, None)
            self.requests_workers_ids.pop(group_id, None)
            #cur_free_gpus_list.append(free_gpus_list[-1])
        else:
            temp = []
            for i in range(1, len(self.requests_workers_ids[group_id])):
                self.gpu_status[self.requests_workers_ids[group_id][i]] = 0
                temp.append(self.requests_workers_ids[group_id][i])
            print(f"request {group_id} release the workers {temp}")
           
            self.hungry_requests.pop(group_id, None)
            #cur_free_gpus_list.append(free_gpus_list[i])
        print(f"after release {self.gpu_status}")

        if not self.hungry_requests:
            return
        
        temp_sorted_requests = list(self.hungry_requests.values())
        temp_sorted_requests.sort(key = lambda x: (self.requests_cur_steps[x.request_id] - self.requests_last_steps[x.request_id]) 
                                  * (self.dit_times[x.resolution][len(self.requests_workers_ids[x.request_id])] - self.dit_times[x.resolution][self.opt_gpus_num[x.resolution]]) 
                                  / self.denoising_steps
                                  , reverse = True)
        
        cur_free_gpus_list = Queue()
        [cur_free_gpus_list.put(i) for i in range(len(self.gpu_status)) if self.gpu_status[i] == 0]
        free_gpu_num = cur_free_gpus_list.qsize()
        
        '''temp_max_gpus_num: Dict[str, int] = {}
        temp_sorted_requests = []
        temp_requests_cur_steps: Dict[str, int] = {}
        for _, seq_group in self.hungry_requests.items():
            cur_workers_num = len(self.requests_workers_ids[seq_group.request_id])
            if self.opt_gpus_num[seq_group.resolution] == 4:
                if cur_workers_num + free_gpu_num >= 4:
                    temp_max_gpus_num[seq_group.request_id] = 4
                    temp_sorted_requests.append(seq_group)
                    temp_requests_cur_steps[seq_group.request_id] = self.requests_cur_steps[seq_group.request_id]
                elif cur_workers_num + free_gpu_num == 2 or cur_workers_num + free_gpu_num == 3:
                    temp_max_gpus_num[seq_group.request_id] = 2
                    temp_sorted_requests.append(seq_group)
                    temp_requests_cur_steps[seq_group.request_id] = self.requests_cur_steps[seq_group.request_id]
            elif self.opt_gpus_num[seq_group.resolution] == 2:
                if cur_workers_num + free_gpu_num >= 2:
                    temp_max_gpus_num[seq_group.request_id] = 2
                    temp_sorted_requests.append(seq_group)
                    temp_requests_cur_steps[seq_group.request_id] = self.requests_cur_steps[seq_group.request_id]
        
        for seq_group in self.waiting:  
            if self.opt_gpus_num[seq_group.resolution] == 4:
                if free_gpu_num >= 4:
                    temp_max_gpus_num[seq_group.request_id] = 4
                    temp_sorted_requests.append(seq_group)
                    temp_requests_cur_steps[seq_group.request_id] = 0
                elif free_gpu_num == 2 or free_gpu_num == 3:
                    temp_max_gpus_num[seq_group.request_id] = 2
                    temp_sorted_requests.append(seq_group)
                    temp_requests_cur_steps[seq_group.request_id] = 0
                elif free_gpu_num == 1:
                    temp_max_gpus_num[seq_group.request_id] = 1
                    temp_sorted_requests.append(seq_group)
                    temp_requests_cur_steps[seq_group.request_id] = 0
            elif self.opt_gpus_num[seq_group.resolution] == 2:
                if free_gpu_num >= 2:
                    temp_max_gpus_num[seq_group.request_id] = 2
                    temp_sorted_requests.append(seq_group)
                    temp_requests_cur_steps[seq_group.request_id] = 0
                elif free_gpu_num == 1:
                    temp_max_gpus_num[seq_group.request_id] = 1
                    temp_sorted_requests.append(seq_group)
                    temp_requests_cur_steps[seq_group.request_id] = 0
            elif self.opt_gpus_num[seq_group.resolution] == 1:
                if free_gpu_num >= 1:
                    temp_max_gpus_num[seq_group.request_id] = 1
                    temp_sorted_requests.append(seq_group)
                    temp_requests_cur_steps[seq_group.request_id] = 0
        
        temp_sorted_requests.sort(key = lambda x: (1 - temp_requests_cur_steps[x.request_id] / self.denoising_steps)
                                  * self.dit_times[x.resolution][temp_max_gpus_num[x.request_id]])'''

        remove_groups: List[str] = []
        update_groups: List[str] = []
        for seq_group in temp_sorted_requests:
            if cur_free_gpus_list.empty():
                break
            #if seq_group not in self.hungry_requests:
            #    free_gpu_num -= temp_max_gpus_num[seq_group.request_id]
            #    continue
            cur_workers_num = len(self.requests_workers_ids[seq_group.request_id])
            if self.opt_gpus_num[seq_group.resolution] == 4:
                if cur_workers_num + free_gpu_num >= 4:
                    count = 4 - cur_workers_num
                    for i in range(count):
                        id = cur_free_gpus_list.get()
                        self.gpu_status[id] = 1
                        self.requests_workers_ids[seq_group.request_id].append(id)
                    #seq_group.worker_ids = copy.deepcopy(self.requests_workers_ids[seq_group.request_id])
                    free_gpu_num -= count
                    remove_groups.append(seq_group.request_id)
                    update_groups.append(seq_group.request_id)
                    del self.requests_cur_steps[seq_group.request_id]
                    del self.requests_last_steps[seq_group.request_id]
                
                elif cur_workers_num + free_gpu_num == 2 or cur_workers_num + free_gpu_num == 3:
                    count = 2 - cur_workers_num
                    for i in range(count):
                        id = cur_free_gpus_list.get()
                        self.gpu_status[id] = 1
                        self.requests_workers_ids[seq_group.request_id].append(id)
                    #seq_group.worker_ids = copy.deepcopy(self.requests_workers_ids[seq_group.request_id])
                    free_gpu_num -= count
                    update_groups.append(seq_group.request_id)
                    self.requests_last_steps[seq_group.request_id] = self.requests_cur_steps[seq_group.request_id]
            
            elif self.opt_gpus_num[seq_group.resolution] == 2:
                if cur_workers_num + free_gpu_num >= 2:
                    count = 2 - cur_workers_num
                    for i in range(count):
                        id = cur_free_gpus_list.get()
                        self.gpu_status[id] = 1
                        self.requests_workers_ids[seq_group.request_id].append(id)
                    #seq_group.worker_ids = copy.deepcopy(self.requests_workers_ids[seq_group.request_id])
                    free_gpu_num -= count
                    remove_groups.append(seq_group.request_id)
                    update_groups.append(seq_group.request_id)
                    del self.requests_cur_steps[seq_group.request_id]
                    del self.requests_last_steps[seq_group.request_id]

        for group_id in update_groups:
            print(f"request {group_id} new workers ids {self.requests_workers_ids[group_id]}")
            self.update_tasks.put((group_id, self.requests_workers_ids[group_id]))
        
        for group_id in remove_groups:
            del self.hungry_requests[group_id]
            #del self.requests_workers_ids[group_id]
        
        print(f"after update {self.gpu_status}")

    #async def schedule
    def schedule(self) -> SequenceGroup:
        #async with self.gpu_status_lock:
        if self.waiting:
            seq_group = self.waiting[0]
            temp_worker_ids = [i for i in range(len(self.gpu_status)) if self.gpu_status[i] == 0]
            
            #print(f"before schedule {self.gpu_status}")
            if seq_group.resolution == "360p":
                if len(temp_worker_ids) >= 4:
                    worker_ids = [temp_worker_ids[i] for i in range(4)]
                    for id in worker_ids:
                        self.gpu_status[id] = 1
                    seq_group.worker_ids = copy.deepcopy(worker_ids)
                    self.requests_workers_ids[seq_group.request_id] = copy.deepcopy(worker_ids)
                    self.waiting.popleft()
                    return seq_group
                
                elif len(temp_worker_ids) == 2 or len(temp_worker_ids) == 3:
                    worker_ids = [temp_worker_ids[i] for i in range(2)]
                    for id in worker_ids:
                        self.gpu_status[id] = 1
                    seq_group.worker_ids = copy.deepcopy(worker_ids)
                    self.hungry_requests[seq_group.request_id] = seq_group
                    self.requests_workers_ids[seq_group.request_id] = copy.deepcopy(worker_ids)
                    self.requests_cur_steps[seq_group.request_id] = 0
                    self.requests_last_steps[seq_group.request_id] = 0
                    self.waiting.popleft()
                    return seq_group
                
                elif len(temp_worker_ids) == 1:
                    worker_ids = [temp_worker_ids[0]]
                    self.gpu_status[temp_worker_ids[0]] = 1
                    seq_group.worker_ids = copy.deepcopy(worker_ids)
                    self.hungry_requests[seq_group.request_id] = seq_group
                    self.requests_workers_ids[seq_group.request_id] = copy.deepcopy(worker_ids)
                    self.requests_cur_steps[seq_group.request_id] = 0
                    self.requests_last_steps[seq_group.request_id] = 0
                    self.waiting.popleft()
                    return seq_group
            
            elif seq_group.resolution == "240p":
                if len(temp_worker_ids) >= 2:
                    worker_ids = [temp_worker_ids[i] for i in range(2)]
                    for id in worker_ids:
                        self.gpu_status[id] = 1
                    seq_group.worker_ids = copy.deepcopy(worker_ids)
                    self.requests_workers_ids[seq_group.request_id] = copy.deepcopy(worker_ids)
                    self.waiting.popleft()
                    return seq_group
                
                elif len(temp_worker_ids) == 1:
                    worker_ids = [temp_worker_ids[0]]
                    self.gpu_status[temp_worker_ids[0]] = 1
                    seq_group.worker_ids = copy.deepcopy(worker_ids)
                    self.hungry_requests[seq_group.request_id] = seq_group
                    self.requests_workers_ids[seq_group.request_id] = copy.deepcopy(worker_ids)
                    self.requests_cur_steps[seq_group.request_id] = 0
                    self.requests_last_steps[seq_group.request_id] = 0
                    self.waiting.popleft()
                    return seq_group
            
            elif seq_group.resolution == "144p":
                if len(temp_worker_ids) >= 1:
                    worker_ids = [temp_worker_ids[0]]
                    self.gpu_status[temp_worker_ids[0]] = 1
                    seq_group.worker_ids = copy.deepcopy(worker_ids)
                    self.requests_workers_ids[seq_group.request_id] = copy.deepcopy(worker_ids)
                    self.waiting.popleft()
                    return seq_group
            #print(f"after schedule {self.gpu_status}")
        
        return None

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)
    
class Scheduler:
    def __init__(
            self,
        ) -> None:
            # Sequence groups in the WAITING state.
            self.waiting: Deque[SequenceGroup] = deque()
            
            self.send_transfering: Dict[str, SequenceGroup] = {}
            self.recv_transfering: Dict[str, SequenceGroup] = {}
                
            self.vae_waiting: Deque[Tuple[SequenceGroup, List]] = deque(tuple())
            
            self.send_finished_req_ids: List[str] = []
            self.recv_finished_req_ids: List[str] = []

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def add_vae_seq_group(self, seq_group: SequenceGroup) -> None:
        self.vae_waiting.append(seq_group)

    def add_send_transfering(self, seq_group: SequenceGroup) -> None:
        self.send_transfering[seq_group.request_id] = seq_group
    
    def del_send_transfering(self, request_id: str) -> None:
        # Delete sequence groups to the send  transfering map 
        if request_id in self.send_transfering:
            del self.send_transfering[request_id]
    
    def add_recv_transfering(self, seq_group: SequenceGroup) -> None:
        #Add sequence groups to the recv transfering map
        self.recv_transfering[seq_group.request_id] = seq_group
        
    def add_send_finished(self, request_ids: List[str]):
        self.send_finished_req_ids.extend(request_ids)
        
    def add_recv_finished(self, request_ids: List[str]):
        self.recv_finished_req_ids.extend(request_ids)
        
    def schedule(self) -> SequenceGroup:
        if self.waiting:
            seq_group = self.waiting[0]
            self.waiting.popleft()
            return seq_group
        return None
    
    #kv缓存传输完了
    def _check_tranfer_finished_req(self) -> None:
        send_finished = []
        for request_id in self.send_finished_req_ids[:]:
            del self.send_transfering[request_id]
            self.send_finished_req_ids.remove(request_id)
            send_finished.append(request_id)
            
        for request_id in self.recv_finished_req_ids[:]:
            seq_group = self.recv_transfering[request_id]
            del self.recv_transfering[request_id]
            self.recv_finished_req_ids.remove(request_id)
            self.waiting.append(seq_group)
        return send_finished