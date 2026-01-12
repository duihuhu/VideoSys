from typing import Deque, Dict, Tuple, List, Optional
from collections import deque
from videosys.core.sequence import SequenceGroup
import requests
import copy
import threading
from queue import Queue

class VideoScheduler:
    def __init__(
        self,
        instances_num: int,
        static_dop: Optional[int] = 8,
        window_size: Optional[int] = 10
        ) -> None:
        # Sequence groups in the WAITING state.
        #self.waiting: Deque[SequenceGroup] = deque()
        self.waiting: Dict[str, SequenceGroup] = {}
        self.num_gpus = 2
        self.num = 0
        
        self.gpu_status = [0 for _ in range(instances_num)]
        #self.gpu_status_lock = asyncio.Lock()
        
        self.hungry_requests: Dict[str, SequenceGroup] = {}
        self.requests_workers_ids: Dict[str, List[int]] = {}
        self.requests_last_steps: Dict[str, int] = {}
        self.requests_cur_steps: Dict[str, int] = {}
        
        '''self.dit_times: Dict[str, Dict[int, float]] = {"144p": {1: 3, 2: 3.4, 4: 3.5}, 
                                                       "240p": {1: 8.3, 2: 4.6, 4: 3.7}, 
                                                       "360p": {1: 19.2, 2: 10.4, 4: 6.1}}'''
        '''self.dit_times: Dict[str, Dict[int, float]] = {"144p": {1: 2.63, 2: 2.05, 4: 2.10, 8: 2.17}, 
                                                       "240p": {1: 6.66, 2: 3.21, 4: 2.17, 8: 2.24}, 
                                                       "360p": {1: 14.31, 2: 6.66, 4: 3.73, 8: 2.23}}'''
        self.dit_times: Dict[str, Dict[int, float]] = {"144p": {1: 4.45, 2: 3.67, 4: 3.69, 8: 3.98}, 
                                                       "240p": {1: 11.27, 2: 5.88, 4: 3.67, 8: 4.10}, 
                                                       "360p": {1: 25.05, 2: 11.91, 4: 6.68, 8: 4.16},
                                                       "480p": {1: 44.24, 2: 20.43, 4: 11.12, 8: 6.31},
                                                       "720p": {1: 112.76, 2: 48.64, 4: 25.08, 8: 13.39},
                                                       "1080p": {1: 311.17, 2: 126.06, 4: 65.01, 8: 33.13},
                                                       "2k": {1: 664.09, 2: 263.21, 4: 134.25, 8: 68.32}}
        
        self.vae_times: Dict[str, float] = {"144p": 0.34, "240p": 0.78, "360p": 1.81, "480p": 3.54, "720p": 8.70, "1080p": 22.04, "2k": 45.36}
        
        #self.opt_gpus_num: Dict[str, int] = {"144p": 1, "240p": 2, "360p": 2, "480p": 4}
        self.opt_gpus_num: Dict[str, int] = {"144p": 1, "240p": 2, "360p": 8, "480p": 8, "720p": 8, "1080p": 8, "2k": 8}

        self.denoising_steps = 30

        self.update_tasks: Queue[Tuple[str, List[int]]] = Queue()
        self.async_server_url = "http://127.0.0.1:8000/request_workers"

        self.static_dop = static_dop #1, 2, 4, 8 add for test
        #self.w1_num: List[Tuple[int, int]] = [(0, 1), (2, 3)]
        #self.w2_num: List[Tuple[int, int]] = [(4, 5)]
        #self.w3_num: List[Tuple[int, int]] = [(6, 7)]
        self.w1_num: List[int] = [0, 1]
        self.w2_num: List[int] = [2, 3, 4]
        self.w3_num: List[int] = [5, 6, 7]

        self.r1_num: List[int] = [0, 1]
        self.r2_num: List[Tuple[int, int]] = [(2, 3)]
        self.r3_num: List[Tuple[int, int]] = [(4, 5), (6, 7)]

        self.home_town: Dict[str, int] = {}

        self.window_size = window_size
    
    def post_http_request(self, pload, api_url) -> requests.Response:
        headers = {"User-Agent": "Test Client"}
        response = requests.post(api_url, headers = headers, json = pload)
        return response

    def update_requests_cur_steps(self, request_id: str, cur_step: int) -> None:
        if request_id in self.requests_cur_steps:
            self.requests_cur_steps[request_id] = cur_step
    
    def update_dit_workers_ids_for_request(self) -> None:
        while True:
            #if self.update_tasks.empty():
            #    continue
            group_id, worker_ids = self.update_tasks.get()
            #add for debug
            '''if len(worker_ids) % 2 != 0:
                self.gpu_status[worker_ids[-1]] = 0
                self.requests_workers_ids[group_id].pop()
                worker_ids.pop()'''
            pload = {
                "request_id": group_id,
                "worker_ids": worker_ids,
            }
            _ = self.post_http_request(pload, self.async_server_url)
           
    def create_update_threads(self) -> None:
        cur_thread = threading.Thread(target = self.update_dit_workers_ids_for_request)
        cur_thread.daemon = True # kill gs -> kill AsyncSched -> Kill VideoSched -> Kill VideoScheduler -> Kill thread
        cur_thread.start()

    def update_gpu_status(self, last: bool, group_id: str, sjf: bool) -> None:
        #for gpu_id in self.requests_workers_ids[group_id]:
        #    self.gpu_status[gpu_id] = 0
        if last:
            #for gpu_id in self.requests_workers_ids[group_id]:
            #    self.gpu_status[gpu_id] = 0
            self.gpu_status[self.requests_workers_ids[group_id][0]] = 0
            self.requests_workers_ids.pop(group_id, None)
        else:
            self.hungry_requests.pop(group_id, None)
            for i in range(1, len(self.requests_workers_ids[group_id])):
                self.gpu_status[self.requests_workers_ids[group_id][i]] = 0
            #for gpu_id in self.requests_workers_ids[group_id][1: ]:
            #    self.gpu_status[gpu_id] = 0
            self.requests_cur_steps.pop(group_id, None)
            if not sjf:
                self.requests_last_steps.pop(group_id, None)
    
    def hungry_first_priority_schedule(self) -> SequenceGroup:
        cur_free_gpus: Queue[int] = Queue()
        [cur_free_gpus.put(gpu_id) for gpu_id, status in enumerate(self.gpu_status) if status == 0]
        if cur_free_gpus.qsize() < 1:
            return None
        #----------process hungry queue in starvation descending order while num = N#
        temp_hungry_requests = list(self.hungry_requests.values())
        # sort in descending order by starvation time
        temp_hungry_requests.sort(key = lambda x: (self.requests_cur_steps[x.request_id] - self.requests_last_steps[x.request_id])
                                    * (self.dit_times[x.resolution][len(self.requests_workers_ids[x.request_id])] - self.dit_times[x.resolution][self.opt_gpus_num[x.resolution]])
                                    / self.denoising_steps
                                    , reverse = True)
        for cur_hungry_request in temp_hungry_requests:
            if cur_free_gpus.qsize() < 1:
                break
            cur_wanted_gpus_num = 1 << (min(cur_free_gpus.qsize() + len(self.requests_workers_ids[cur_hungry_request.request_id]), self.opt_gpus_num[cur_hungry_request.resolution]).bit_length() - 1)
            cur_demand_gpus_num = cur_wanted_gpus_num - len(self.requests_workers_ids[cur_hungry_request.request_id])
            for _ in range(cur_demand_gpus_num):
                gpu_id = cur_free_gpus.get()
                self.gpu_status[gpu_id] = 1
                self.requests_workers_ids[cur_hungry_request.request_id].append(gpu_id)
            self.update_tasks.put((cur_hungry_request.request_id, self.requests_workers_ids[cur_hungry_request.request_id]))
            if cur_wanted_gpus_num == self.opt_gpus_num[cur_hungry_request.resolution]:
                self.hungry_requests.pop(cur_hungry_request.request_id, None)
                self.requests_cur_steps.pop(cur_hungry_request.request_id, None)
                self.requests_last_steps.pop(cur_hungry_request.request_id, None)
            else:
                #if cur_hungry_request.request_id in self.requests_last_steps:
                self.requests_last_steps[cur_hungry_request.request_id] = self.requests_cur_steps[cur_hungry_request.request_id]
        #----------process waiting queue in FCFS while num = 1----------#
        if self.waiting:
            if cur_free_gpus.qsize() < 1:
                return None
            cur_request_id, cur_waiting_request = list(self.waiting.items())[0]
            cur_wanted_gpus_num = 1 << (min(cur_free_gpus.qsize(), self.opt_gpus_num[cur_waiting_request.resolution]).bit_length() - 1)
            for _ in range(cur_wanted_gpus_num):
                gpu_id = cur_free_gpus.get()
                self.gpu_status[gpu_id] = 1
                if cur_request_id not in self.requests_workers_ids:
                    self.requests_workers_ids[cur_request_id] = [gpu_id]
                else:
                    self.requests_workers_ids[cur_request_id].append(gpu_id)
            if cur_wanted_gpus_num < self.opt_gpus_num[cur_waiting_request.resolution]:
                self.hungry_requests[cur_request_id] = cur_waiting_request
                self.requests_cur_steps[cur_request_id] = 0
                self.requests_last_steps[cur_request_id] = 0
            cur_waiting_request.worker_ids = copy.deepcopy(self.requests_workers_ids[cur_request_id])
            self.waiting.pop(cur_request_id, None)
            return cur_waiting_request
        return None
    
    def continuous_batching_schedule(self) -> SequenceGroup:
        cur_free_gpus: Queue[int] = Queue()
        for gpu_id, status in enumerate(self.gpu_status):
            if status == 0:
                cur_free_gpus.put(gpu_id)
        if cur_free_gpus.qsize() < 1:
            return None
        
        if self.waiting:
            cur_waiting_request = self.waiting[0]
            cur_demand_gpus_num = []
            cur_max_gpus_num = self.opt_gpus_num[cur_waiting_request.resolution]
            while cur_max_gpus_num > 0:
                cur_demand_gpus_num.append(cur_max_gpus_num)
                cur_max_gpus_num //= 2
            for demand_gpus_num in cur_demand_gpus_num:
                if cur_free_gpus.qsize() < demand_gpus_num:
                    continue
                for _ in range(demand_gpus_num):
                    gpu_id = cur_free_gpus.get()
                    self.gpu_status[gpu_id] = 1
                    if cur_waiting_request.request_id not in self.requests_workers_ids:
                        self.requests_workers_ids[cur_waiting_request.request_id] = [gpu_id]
                    else:
                        self.requests_workers_ids[cur_waiting_request.request_id].append(gpu_id)
                cur_waiting_request.worker_ids = copy.deepcopy(self.requests_workers_ids[cur_waiting_request.request_id])
                self.waiting.popleft()
                return cur_waiting_request
        
        return None
    
    def breakdown_update_gpu_status(self, group_id: str, last: bool) -> None:
        if last:
            self.gpu_status[self.requests_workers_ids[group_id][0]] = 0
            self.requests_workers_ids.pop(group_id, None)
        else:
            for i in range(1, len(self.requests_workers_ids[group_id])):
                self.gpu_status[self.requests_workers_ids[group_id][i]] = 0
        
    def naive_update_gpu_status(self, group_id: str) -> None:
        for gpu_id in self.requests_workers_ids[group_id]:
            self.gpu_status[gpu_id] = 0
        self.requests_workers_ids.pop(group_id, None)
    
    def window_update_gpu_status(self, group_id: str) -> None:
        self.hungry_requests.pop(group_id, None)
        self.requests_cur_steps.pop(group_id, None)
        #self.requests_last_steps.pop(group_id, None) 
        #no for sjf
        for gpu_id in self.requests_workers_ids[group_id]:
            self.gpu_status[gpu_id] = 0
        self.requests_workers_ids.pop(group_id, None)
        
    def naive_baseline_schedule(self) -> SequenceGroup:
        cur_free_gpus: Queue[int] = Queue()
        [cur_free_gpus.put(gpu_id) for gpu_id, status in enumerate(self.gpu_status) if status == 0]
        availible_gpus_num = cur_free_gpus.qsize()
        '''for gpu_id, status in enumerate(self.gpu_status):
            if status == 0:
                cur_free_gpus.put(gpu_id)
        '''
        if availible_gpus_num < 1:
            return None
        if self.waiting:
            cur_request_id, cur_waiting_request = list(self.waiting.items())[0]
            if availible_gpus_num < self.opt_gpus_num[cur_waiting_request.resolution]:
                return None
            for _ in range(self.opt_gpus_num[cur_waiting_request.resolution]):
                gpu_id = cur_free_gpus.get()
                self.gpu_status[gpu_id] = 1
                if cur_request_id not in self.requests_workers_ids:
                    self.requests_workers_ids[cur_request_id] = [gpu_id]
                else:
                    self.requests_workers_ids[cur_request_id].append(gpu_id)
            cur_waiting_request.worker_ids = copy.deepcopy(self.requests_workers_ids[cur_request_id])
            #self.waiting.popleft()
            self.waiting.pop(cur_request_id, None)
            return cur_waiting_request
        return None
    
    def naive_baseline_greedy_schedule(self) -> SequenceGroup:
        cur_free_gpus: Queue[int] = Queue()
        [cur_free_gpus.put(gpu_id) for gpu_id, status in enumerate(self.gpu_status) if status == 0]
        availible_gpus_num = cur_free_gpus.qsize()
        if availible_gpus_num < 1:
            return None
        if self.waiting:
            cur_request_id, cur_waiting_request = list(self.waiting.items())[0]
            possible_gpus_num = 1 << (min(availible_gpus_num, self.opt_gpus_num[cur_waiting_request.resolution]).bit_length() - 1)
            for _ in range(possible_gpus_num):
                gpu_id = cur_free_gpus.get()
                self.gpu_status[gpu_id] = 1
                if cur_request_id not in self.requests_workers_ids:
                    self.requests_workers_ids[cur_request_id] = [gpu_id]
                else:
                    self.requests_workers_ids[cur_request_id].append(gpu_id)
            cur_waiting_request.worker_ids = copy.deepcopy(self.requests_workers_ids[cur_request_id])
            self.waiting.pop(cur_request_id, None)
            return cur_waiting_request
        return None
    
    def naive_baseline_update_gpu_status(self, resolution: str, worker_ids: List[int]) -> None:
        if resolution == '144p':
            #self.w1_num.append((worker_ids[0], worker_ids[1]))
            self.w1_num.append((worker_ids[0]))
        elif resolution == '240p':
            #self.w2_num.append((worker_ids[0], worker_ids[1]))
            self.w2_num.append((worker_ids[0]))
        elif resolution == '360p':
            #self.w3_num.append((worker_ids[0], worker_ids[1]))
            self.w3_num.append((worker_ids[0]))

    def naive_partition_schedule(self) -> SequenceGroup:
        if self.waiting:
            cur_req = self.waiting[0]
            if cur_req.resolution == '144p' and len(self.w1_num) >= 1:
                #x, y = self.w1_num.pop(0)
                x = self.w1_num.pop(0)
                #temp_worker_ids = [x, y]
                temp_worker_ids = [x]
                cur_req.worker_ids = temp_worker_ids
                self.waiting.popleft()
                return cur_req
            elif cur_req.resolution == '240p' and len(self.w2_num) >= 1:
                #x, y = self.w2_num.pop(0)
                x = self.w2_num.pop(0)
                #temp_worker_ids = [x, y]
                temp_worker_ids = [x]
                cur_req.worker_ids = temp_worker_ids
                self.waiting.popleft()
                return cur_req
            elif cur_req.resolution == '360p' and len(self.w3_num) >= 1:
                #x, y = self.w3_num.pop(0)
                x = self.w3_num.pop(0)
                #temp_worker_ids = [x, y]
                temp_worker_ids = [x]
                cur_req.worker_ids = temp_worker_ids
                self.waiting.popleft()
                return cur_req
        return None

    def smart_baseline_update_gpu_status(self, worker_ids: List[int], res: str) -> None:
        if len(worker_ids) == 1:
            self.r1_num.append(worker_ids[0])
            #self.home_town.pop(req_id, None)
        elif len(worker_ids) == 2:
            if res == '240p': #self.home_town[req_id] == 1:
                self.r2_num.append((worker_ids[0], worker_ids[1]))
            elif res == '360p': #self.home_town[req_id] == 2:
                self.r3_num.append((worker_ids[0], worker_ids[1]))
            #self.home_town.pop(req_id, None)
        #elif len(worker_ids) == 4:
        #    self.r3_num.append((worker_ids[0], worker_ids[1], worker_ids[2], worker_ids[3]))
    
    def smart_static_partition_schedule(self) -> SequenceGroup:
        if self.waiting:
            cur_req = self.waiting[0]
            if cur_req.resolution == '144p' and len(self.r1_num) >= 1:
                x = self.r1_num.pop(0)
                temp_worker_ids = [x]
                cur_req.worker_ids = temp_worker_ids
                self.waiting.popleft()
                return cur_req
            elif cur_req.resolution == '240p' and len(self.r2_num) >= 1:
                x, y = self.r2_num.pop(0)
                temp_worker_ids = [x, y]
                cur_req.worker_ids = temp_worker_ids
                self.waiting.popleft()
                return cur_req
            elif cur_req.resolution == '360p' and len(self.r3_num) >= 1:
                x, y = self.r3_num.pop(0)
                temp_worker_ids = [x, y]
                cur_req.worker_ids = temp_worker_ids
                self.waiting.popleft()
                return cur_req
        return None
    
    def smart_dynamic_partition_schedule(self) -> SequenceGroup:
        if self.waiting:
            cur_req = self.waiting[0]
            if cur_req.resolution == '144p':
                if len(self.r1_num) >= 1:
                    x = self.r1_num.pop(0)
                    temp_worker_ids = [x]
                    cur_req.worker_ids = temp_worker_ids
                    self.home_town[cur_req.request_id] = 0
                    self.waiting.popleft()
                    return cur_req
                elif len(self.r3_num) >= 1:
                    x, y = self.r3_num.pop(0)
                    temp_worker_ids = [x, y]
                    cur_req.worker_ids = temp_worker_ids
                    self.home_town[cur_req.request_id] = 2
                    self.waiting.popleft()
                    return cur_req
                elif len(self.r2_num) >= 1:
                    x, y = self.r2_num.pop(0)
                    temp_worker_ids = [x, y]
                    cur_req.worker_ids = temp_worker_ids
                    self.home_town[cur_req.request_id] = 1
                    self.waiting.popleft()
                    return cur_req
            elif cur_req.resolution == '240p':
                if len(self.r2_num) >= 1:
                    x, y = self.r2_num.pop(0)
                    temp_worker_ids = [x, y]
                    cur_req.worker_ids = temp_worker_ids
                    self.home_town[cur_req.request_id] = 1
                    self.waiting.popleft()
                    return cur_req
                elif len(self.r3_num) >= 1:
                    x, y = self.r3_num.pop(0)
                    temp_worker_ids = [x, y]
                    cur_req.worker_ids = temp_worker_ids
                    self.home_town[cur_req.request_id] = 2
                    self.waiting.popleft()
                    return cur_req
                elif len(self.r1_num) >= 1:
                    x = self.r1_num.pop(0)
                    temp_worker_ids = [x]
                    cur_req.worker_ids = temp_worker_ids
                    self.home_town[cur_req.request_id] = 0
                    self.waiting.popleft()
                    return cur_req
            elif cur_req.resolution == '360p':
                if len(self.r3_num) >= 1:
                    x, y = self.r3_num.pop(0)
                    temp_worker_ids = [x, y]
                    cur_req.worker_ids = temp_worker_ids
                    self.home_town[cur_req.request_id] = 2
                    self.waiting.popleft()
                    return cur_req
                elif len(self.r2_num) >= 1:
                    x, y = self.r2_num.pop(0)
                    temp_worker_ids = [x, y]
                    cur_req.worker_ids = temp_worker_ids
                    self.home_town[cur_req.request_id] = 1
                    self.waiting.popleft()
                    return cur_req
                elif len(self.r1_num) >= 1:
                    x = self.r1_num.pop(0)
                    temp_worker_ids = [x]
                    cur_req.worker_ids = temp_worker_ids
                    self.home_town[cur_req.request_id] = 0
                    self.waiting.popleft()
                    return cur_req
        return None
    
    def fcfs_decouple_sjf_schedule(self) -> SequenceGroup:
        cur_free_gpus: Queue[int] = Queue()
        [cur_free_gpus.put(gpu_id) for gpu_id, status in enumerate(self.gpu_status) if status == 0]
        if cur_free_gpus.qsize() < 1:
            return None
        
        temp_requests_list: List[SequenceGroup] = []
        temp_remaining_times: Dict[str, float] = {}
        temp_requests_max_gpus_num: Dict[str, int] = {}

        if self.hungry_requests:
            for request_id, cur_seq_group in self.hungry_requests.items():
                temp_requests_list.append(cur_seq_group)
                cur_max_gpus_num = 1 << (min(cur_free_gpus.qsize() + len(self.requests_workers_ids[request_id]), self.opt_gpus_num[cur_seq_group.resolution]).bit_length() - 1)
                temp_requests_max_gpus_num[request_id] = cur_max_gpus_num
                temp_remaining_times[request_id] = self.dit_times[cur_seq_group.resolution][cur_max_gpus_num] * (1 - (self.requests_cur_steps[request_id] / self.denoising_steps))

        if self.waiting:
            id, seq = list(self.waiting.items())[0]
            temp_requests_list.append(seq)
            temp_max_gpus_num = 1 << (min(cur_free_gpus.qsize(), self.opt_gpus_num[seq.resolution]).bit_length() - 1)
            temp_requests_max_gpus_num[id] = temp_max_gpus_num
            temp_remaining_times[id] = self.dit_times[seq.resolution][temp_max_gpus_num]
        
        if temp_requests_list:
            temp_requests_list.sort(key = lambda x: temp_remaining_times[x.request_id])
            
            ans = -1
            for i, this_seq_group in enumerate(temp_requests_list): 
                if cur_free_gpus.qsize() < 1:
                    break
                if this_seq_group.request_id in self.waiting:
                    if cur_free_gpus.qsize() < temp_requests_max_gpus_num[this_seq_group.request_id]:
                        continue
                    for _ in range(temp_requests_max_gpus_num[this_seq_group.request_id]):
                        gpu_id = cur_free_gpus.get()
                        self.gpu_status[gpu_id] = 1
                        if this_seq_group.request_id not in self.requests_workers_ids:
                            self.requests_workers_ids[this_seq_group.request_id] = [gpu_id]
                        else:
                            self.requests_workers_ids[this_seq_group.request_id].append(gpu_id)
                    if len(self.requests_workers_ids[this_seq_group.request_id]) < self.opt_gpus_num[this_seq_group.resolution]:
                        self.hungry_requests[this_seq_group.request_id] = this_seq_group
                        self.requests_cur_steps[this_seq_group.request_id] = 0
                    this_seq_group.worker_ids = copy.deepcopy(self.requests_workers_ids[this_seq_group.request_id])
                    self.waiting.pop(this_seq_group.request_id, None)
                    ans = i
                else:
                    temp_allocated_gpus_num = temp_requests_max_gpus_num[this_seq_group.request_id] - len(self.requests_workers_ids[this_seq_group.request_id])
                    if cur_free_gpus.qsize() < temp_allocated_gpus_num:
                        continue
                    for _ in range(temp_allocated_gpus_num):
                        gpu_id = cur_free_gpus.get()
                        self.gpu_status[gpu_id] = 1
                        self.requests_workers_ids[this_seq_group.request_id].append(gpu_id)
                    self.update_tasks.put((this_seq_group.request_id, self.requests_workers_ids[this_seq_group.request_id]))
                    if temp_requests_max_gpus_num[this_seq_group.request_id] == self.opt_gpus_num[this_seq_group.resolution]:
                        self.hungry_requests.pop(this_seq_group.request_id, None)
                        self.requests_cur_steps.pop(this_seq_group.request_id, None)
            return temp_requests_list[ans] if ans != -1 else None
        return None
    
    def window_based_sjf_schedule(self) -> SequenceGroup:
        cur_free_gpus: Queue[int] = Queue()
        [cur_free_gpus.put(gpu_id) for gpu_id, status in enumerate(self.gpu_status) if status == 0]
        if cur_free_gpus.qsize() < 1:
            return None
        
        if self.waiting:
            window_seqs: List[Tuple[str, SequenceGroup]] = list(self.waiting.items())[: self.window_size]
            temp_remaing_times: Dict[str, float] = {}
            for w_request_id, w_seq in window_seqs:
                temp_max_gpus_num = 1 << (min(cur_free_gpus.qsize(), self.opt_gpus_num[w_seq.resolution]).bit_length() - 1)
                temp_remaing_times[w_request_id] = self.dit_times[w_seq.resolution][temp_max_gpus_num]
            window_seqs.sort(key = lambda x: temp_remaing_times[x[0]])

            cur_request_id, cur_seq_group = window_seqs[0]
            temp_cur_gpus_num = 1 << (min(cur_free_gpus.qsize(), self.opt_gpus_num[cur_seq_group.resolution]).bit_length() - 1)
            for _ in range(temp_cur_gpus_num):
                gpu_id = cur_free_gpus.get()
                self.gpu_status[gpu_id] = 1
                if cur_request_id not in self.requests_workers_ids:
                    self.requests_workers_ids[cur_request_id] = [gpu_id]
                else:
                    self.requests_workers_ids[cur_request_id].append(gpu_id)
            cur_seq_group.worker_ids = copy.deepcopy(self.requests_workers_ids[cur_request_id])
            self.waiting.pop(cur_request_id, None)
            return cur_seq_group
        
        return None

    def window_based_sjf_with_hungry_update_schedule(self) -> SequenceGroup:
        cur_free_gpus: Queue[int] = Queue()
        [cur_free_gpus.put(gpu_id) for gpu_id, status in enumerate(self.gpu_status) if status == 0]
        if cur_free_gpus.qsize() < 1:
            return None
        #----------process hungry queue in starvation descending order while num = N#
        temp_hungry_requests = list(self.hungry_requests.values())
        # sort in descending order by starvation time
        temp_hungry_requests.sort(key = lambda x: (self.requests_cur_steps[x.request_id] - self.requests_last_steps[x.request_id])
                                    * (self.dit_times[x.resolution][len(self.requests_workers_ids[x.request_id])] - self.dit_times[x.resolution][self.opt_gpus_num[x.resolution]])
                                    / self.denoising_steps
                                    , reverse = True)
        for cur_hungry_request in temp_hungry_requests:
            if cur_free_gpus.qsize() < 1:
                break
            cur_wanted_gpus_num = 1 << (min(cur_free_gpus.qsize() + len(self.requests_workers_ids[cur_hungry_request.request_id]), self.opt_gpus_num[cur_hungry_request.resolution]).bit_length() - 1)
            cur_demand_gpus_num = cur_wanted_gpus_num - len(self.requests_workers_ids[cur_hungry_request.request_id])
            for _ in range(cur_demand_gpus_num):
                gpu_id = cur_free_gpus.get()
                self.gpu_status[gpu_id] = 1
                self.requests_workers_ids[cur_hungry_request.request_id].append(gpu_id)
            self.update_tasks.put((cur_hungry_request.request_id, self.requests_workers_ids[cur_hungry_request.request_id]))
            if cur_wanted_gpus_num == self.opt_gpus_num[cur_hungry_request.resolution]:
                self.hungry_requests.pop(cur_hungry_request.request_id, None)
                self.requests_cur_steps.pop(cur_hungry_request.request_id, None)
                self.requests_last_steps.pop(cur_hungry_request.request_id, None)
            else:
                #if cur_hungry_request.request_id in self.requests_last_steps:
                self.requests_last_steps[cur_hungry_request.request_id] = self.requests_cur_steps[cur_hungry_request.request_id]
        #----------process waiting queue in FCFS while num = 1----------#
        if self.waiting:
            if cur_free_gpus.qsize() < 1:
                return None
            window_seqs: List[Tuple[str, SequenceGroup]] = list(self.waiting.items())[: self.window_size]
            temp_remaing_times: Dict[str, float] = {}
            for w_request_id, w_seq in window_seqs:
                temp_max_gpus_num = 1 << (min(cur_free_gpus.qsize(), self.opt_gpus_num[w_seq.resolution]).bit_length() - 1)
                temp_remaing_times[w_request_id] = self.dit_times[w_seq.resolution][temp_max_gpus_num]
            window_seqs.sort(key = lambda x: temp_remaing_times[x[0]])

            cur_request_id, cur_seq_group = window_seqs[0]
            temp_cur_gpus_num = 1 << (min(cur_free_gpus.qsize(), self.opt_gpus_num[cur_seq_group.resolution]).bit_length() - 1)
            for _ in range(temp_cur_gpus_num):
                gpu_id = cur_free_gpus.get()
                self.gpu_status[gpu_id] = 1
                if cur_request_id not in self.requests_workers_ids:
                    self.requests_workers_ids[cur_request_id] = [gpu_id]
                else:
                    self.requests_workers_ids[cur_request_id].append(gpu_id)
            if temp_cur_gpus_num < self.opt_gpus_num[cur_seq_group.resolution]:
                self.hungry_requests[cur_request_id] = cur_seq_group
                self.requests_cur_steps[cur_request_id] = 0
                self.requests_last_steps[cur_request_id] = 0
            cur_seq_group.worker_ids = copy.deepcopy(self.requests_workers_ids[cur_request_id])
            self.waiting.pop(cur_request_id, None)
            return cur_seq_group
        return None
    
    def window_based_sjf_with_sjf_update_schedule(self) -> List[SequenceGroup]:
        cur_free_gpus: Queue[int] = Queue()
        [cur_free_gpus.put(gpu_id) for gpu_id, status in enumerate(self.gpu_status) if status == 0]
        if cur_free_gpus.qsize() < 1:
            return None
        
        temp_requests_list: List[SequenceGroup] = []
        temp_remaining_times: Dict[str, float] = {}
        temp_requests_max_gpus_num: Dict[str, int] = {}

        if self.hungry_requests:
            for request_id, cur_seq_group in self.hungry_requests.items():
                temp_requests_list.append(cur_seq_group)
                cur_max_gpus_num = 1 << (min(cur_free_gpus.qsize() + len(self.requests_workers_ids[request_id]), self.opt_gpus_num[cur_seq_group.resolution]).bit_length() - 1)
                temp_requests_max_gpus_num[request_id] = cur_max_gpus_num
                temp_remaining_times[request_id] = self.dit_times[cur_seq_group.resolution][cur_max_gpus_num] * (1 - (self.requests_cur_steps[request_id] / self.denoising_steps))

        if self.waiting:
            window_seqs: List[Tuple[str, SequenceGroup]] = list(self.waiting.items())[: self.window_size]
            for id, seq in window_seqs:
                temp_requests_list.append(seq)
                temp_max_gpus_num = 1 << (min(cur_free_gpus.qsize(), self.opt_gpus_num[seq.resolution]).bit_length() - 1)
                temp_requests_max_gpus_num[id] = temp_max_gpus_num
                temp_remaining_times[id] = self.dit_times[seq.resolution][temp_max_gpus_num]
        
        if temp_requests_list:
            temp_requests_list.sort(key = lambda x: temp_remaining_times[x.request_id])
            
            ans = -1
            only_one = True
            for i, this_seq_group in enumerate(temp_requests_list): 
                if cur_free_gpus.qsize() < 1:
                    break
                if this_seq_group.request_id in self.waiting:
                    if cur_free_gpus.qsize() < temp_requests_max_gpus_num[this_seq_group.request_id]:
                        continue
                    for _ in range(temp_requests_max_gpus_num[this_seq_group.request_id]):
                        gpu_id = cur_free_gpus.get()
                        self.gpu_status[gpu_id] = 1
                        if this_seq_group.request_id not in self.requests_workers_ids:
                            self.requests_workers_ids[this_seq_group.request_id] = [gpu_id]
                        else:
                            self.requests_workers_ids[this_seq_group.request_id].append(gpu_id)
                    if len(self.requests_workers_ids[this_seq_group.request_id]) < self.opt_gpus_num[this_seq_group.resolution]:
                        self.hungry_requests[this_seq_group.request_id] = this_seq_group
                        self.requests_cur_steps[this_seq_group.request_id] = 0
                    this_seq_group.worker_ids = copy.deepcopy(self.requests_workers_ids[this_seq_group.request_id])
                    self.waiting.pop(this_seq_group.request_id, None)
                    if only_one:
                        ans = i
                        only_one = False
                else:
                    temp_allocated_gpus_num = temp_requests_max_gpus_num[this_seq_group.request_id] - len(self.requests_workers_ids[this_seq_group.request_id])
                    if cur_free_gpus.qsize() < temp_allocated_gpus_num:
                        continue
                    for _ in range(temp_allocated_gpus_num):
                        gpu_id = cur_free_gpus.get()
                        self.gpu_status[gpu_id] = 1
                        self.requests_workers_ids[this_seq_group.request_id].append(gpu_id)
                    self.update_tasks.put((this_seq_group.request_id, self.requests_workers_ids[this_seq_group.request_id]))
                    if temp_requests_max_gpus_num[this_seq_group.request_id] == self.opt_gpus_num[this_seq_group.resolution]:
                        self.hungry_requests.pop(this_seq_group.request_id, None)
                        self.requests_cur_steps.pop(this_seq_group.request_id, None)
            return temp_requests_list[ans] if ans != -1 else None
        return None
            
    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        #self.waiting.append(seq_group)
        self.waiting[seq_group.request_id] = seq_group
    
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