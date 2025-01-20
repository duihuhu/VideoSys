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

        self.static_dop = 2 #1, 4 add for test
        self.w1: List[Tuple[int, int]] = [(0, 1), (2, 3)]
        self.w2: List[Tuple[int, int]] = [(4, 5)]
        self.w3: List[Tuple[int, int]] = [(6, 7)]

        self.r1: List[int] = [0, 1]
        self.r2: List[Tuple[int, int]] = [(2, 3)]
        self.r3: List[Tuple[int, int, int, int]] = [(4, 5, 6, 7)]
    
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
        if last:
            self.gpu_status[self.requests_workers_ids[group_id][0]] = 0
            self.requests_workers_ids.pop(group_id, None)
        else:
            for i in range(1, len(self.requests_workers_ids[group_id])):
                self.gpu_status[self.requests_workers_ids[group_id][i]] = 0
            self.hungry_requests.pop(group_id, None) 
            self.requests_cur_steps.pop(group_id, None)
            self.requests_last_steps.pop(group_id, None)
    
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
                self.update_tasks.put((cur_hungry_request.request_id, self.requests_workers_ids[cur_hungry_request.request_id]))
                if i == 0:
                    self.hungry_requests.pop(cur_hungry_request.request_id, None)
                    self.requests_cur_steps.pop(cur_hungry_request.request_id, None)
                    self.requests_last_steps.pop(cur_hungry_request.request_id, None)
                else:
                    if cur_hungry_request.request_id in self.requests_last_steps:
                        self.requests_last_steps[cur_hungry_request.request_id] = self.requests_cur_steps[cur_hungry_request.request_id]    
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
                    self.hungry_requests[cur_waiting_request.request_id] = cur_waiting_request
                    self.requests_cur_steps[cur_waiting_request.request_id] = 0
                    self.requests_last_steps[cur_waiting_request.request_id] = 0
                cur_waiting_request.worker_ids = copy.deepcopy(self.requests_workers_ids[cur_waiting_request.request_id])
                self.waiting.popleft()
                return cur_waiting_request
        return None
    
    def navie_update_gpu_status(self, group_id: str) -> None:
        for gpu_id in self.requests_workers_ids[group_id]:
            self.gpu_status[gpu_id] = 0
        self.requests_workers_ids.pop(group_id, None)
    
    def naive_baseline_schedule(self) -> SequenceGroup:
        cur_free_gpus: Queue[int] = Queue()
        for gpu_id, status in enumerate(self.gpu_status):
            if status == 0:
                cur_free_gpus.put(gpu_id)
        if cur_free_gpus.qsize() < self.static_dop:
            return None
        if self.waiting:
            cur_waiting_request = self.waiting[0]
            for _ in range(self.static_dop):
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
    
    def naive_baseline_update_gpu_status(self, resolution: str, worker_ids: List[int]) -> None:
        if resolution == '144p':
            self.w1_num.append((worker_ids[0], worker_ids[1]))
        elif resolution == '240p':
            self.w2_num.append((worker_ids[0], worker_ids[1]))
        elif resolution == '360p':
            self.w3_num.append((worker_ids[0], worker_ids[1]))

    def naive_partition_schedule(self) -> SequenceGroup:
        if self.waiting:
            cur_req = self.waiting[0]
            if cur_req.resolution == '144p' and len(self.w1_num) >= 1:
                x, y = self.w1_num.pop(0)
                temp_worker_ids = [x, y]
                cur_req.worker_ids = temp_worker_ids
                self.waiting.popleft()
                return cur_req
            elif cur_req.resolution == '240p' and len(self.w2_num) >= 1:
                x, y = self.w2_num.pop(0)
                temp_worker_ids = [x, y]
                cur_req.worker_ids = temp_worker_ids
                self.waiting.popleft()
                return cur_req
            elif cur_req.resolution == '360p' and len(self.w3_num) >= 1:
                x, y = self.w3_num.pop(0)
                temp_worker_ids = [x, y]
                cur_req.worker_ids = temp_worker_ids
                self.waiting.popleft()
                return cur_req
        return None

    def smart_baseline_update_gpu_status(self, worker_ids: List[int]) -> None:
        if len(worker_ids) == 1:
            self.r1_num.append(worker_ids[0])
        elif len(worker_ids) == 2:
            self.r2_num.append((worker_ids[0], worker_ids[1]))
        elif len(worker_ids) == 4:
            self.r3_num.append((worker_ids[0], worker_ids[1], worker_ids[2], worker_ids[3]))
    
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
                x, y, z, l = self.r3_num.pop(0)
                temp_worker_ids = [x, y, z, l]
                cur_req.worker_ids = temp_worker_ids
                self.waiting.popleft()
                return cur_req
        return None
    
    def smart_dynamic_partition_schedule(self) -> SequenceGroup:
        if self.waiting:
            cur_req = self.waiting[0]
            if len(self.r1_num) >= 1:
                x = self.r1_num.pop(0)
                temp_worker_ids = [x]
                cur_req.worker_ids = temp_worker_ids
                self.waiting.popleft()
                return cur_req
            elif len(self.r2_num) >= 1:
                x, y = self.r2_num.pop(0)
                temp_worker_ids = [x, y]
                cur_req.worker_ids = temp_worker_ids
                self.waiting.popleft()
                return cur_req
            elif len(self.r3_num) >= 1:
                x, y, z, l = self.r3_num.pop(0)
                temp_worker_ids = [x, y, z, l]
                cur_req.worker_ids = temp_worker_ids
                self.waiting.popleft()
                return cur_req
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