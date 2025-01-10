from typing import Deque, Dict, Tuple, List
from collections import deque
from videosys.core.sequence import SequenceGroup
import requests
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
        
        self.hungry_requests: Dict[str, SequenceGroup] = {}
        self.requests_workers_ids: Dict[str, List[int]] = {}
        self.requests_last_steps: Dict[str, int] = {}
        self.requests_cur_steps: Dict[str, int] = {}
        
        self.dit_times: Dict[str, Dict[int, float]] = {"144p": {1: 3, 2: 3.4, 4: 3.5}, 
                                                       "240p": {1: 8.3, 2: 4.6, 4: 3.7}, 
                                                       "360p": {1: 19.2, 2: 10.4, 4: 6.1}}
        self.opt_gps_num: Dict[str, int] = {"144p": 1, "240p": 2, "360p": 4}
        self.denoising_steps = 30
    
    def post_http_request(self, pload, api_url) -> requests.Response:
        headers = {"User-Agent": "Test Client"}
        response = requests.post(api_url, headers = headers, json = pload)
        return response

    def update_requests_cur_steps(self, request_id: str, cur_step: int) -> None:
        if request_id in self.requests_cur_steps:
            self.requests_cur_steps[request_id] = cur_step

    def update_and_schedule(self, last: bool, free_gpus_list: List[int]):
        #cur_free_gpus_list = []
        if last:
            self.gpu_status[free_gpus_list[-1]] = 0
            #cur_free_gpus_list.append(free_gpus_list[-1])ß
        else:
            for i in range(0, len(free_gpus_list) - 1):
                self.gpu_status[free_gpus_list[i]] = 0
                #cur_free_gpus_list.append(free_gpus_list[i])
        cur_free_gpus_list = [i for i in range(len(self.gpu_status)) if self.gpu_status[i] == 0]
        
        if not self.hungry_requests:
            return
        
        temp_sorted_requests = []
        for _, seq_group in self.hungry_requests.items():
            temp_sorted_requests.append(seq_group)
        temp_sorted_requests.sort(key = lambda x: (self.requests_cur_steps[x.request_id] - self.requests_last_steps[x.request_id]) 
                                  * (self.dit_times[x.resolution][len(x.worker_ids)] - self.dit_times[x.resolution][self.opt_gps_num[x.resolution]]) / self.denoising_steps
                                  , reverse = True)
        
        free_gpu_num = len(cur_free_gpus_list)
        
        '''temp_max_gpus_num: Dict[str, int] = {}
        temp_sorted_requests = []
        temp_requests_cur_steps: Dict[str, int] = {}
        for _, seq_group in self.hungry_requests.items():
            cur_workers_num = len(self.requests_workers_ids[seq_group.request_id])
            if self.opt_gps_num[seq_group.resolution] == 4:
                if cur_workers_num + free_gpu_num >= 4:
                    temp_max_gpus_num[seq_group.request_id] = 4
                    temp_sorted_requests.append(seq_group)
                    temp_requests_cur_steps[seq_group.request_id] = self.requests_cur_steps[seq_group.request_id]
                elif cur_workers_num + free_gpu_num == 2 or cur_workers_num + free_gpu_num == 3:
                    temp_max_gpus_num[seq_group.request_id] = 2
                    temp_sorted_requests.append(seq_group)
                    temp_requests_cur_steps[seq_group.request_id] = self.requests_cur_steps[seq_group.request_id]
            elif self.opt_gps_num[seq_group.resolution] == 2:
                if cur_workers_num + free_gpu_num >= 2:
                    temp_max_gpus_num[seq_group.request_id] = 2
                    temp_sorted_requests.append(seq_group)
                    temp_requests_cur_steps[seq_group.request_id] = self.requests_cur_steps[seq_group.request_id]
        
        for seq_group in self.waiting:  
            if self.opt_gps_num[seq_group.resolution] == 4:
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
            elif self.opt_gps_num[seq_group.resolution] == 2:
                if free_gpu_num >= 2:
                    temp_max_gpus_num[seq_group.request_id] = 2
                    temp_sorted_requests.append(seq_group)
                    temp_requests_cur_steps[seq_group.request_id] = 0
                elif free_gpu_num == 1:
                    temp_max_gpus_num[seq_group.request_id] = 1
                    temp_sorted_requests.append(seq_group)
                    temp_requests_cur_steps[seq_group.request_id] = 0
            elif self.opt_gps_num[seq_group.resolution] == 1:
                if free_gpu_num >= 1:
                    temp_max_gpus_num[seq_group.request_id] = 1
                    temp_sorted_requests.append(seq_group)
                    temp_requests_cur_steps[seq_group.request_id] = 0
        
        temp_sorted_requests.sort(key = lambda x: (1 - temp_requests_cur_steps[x.request_id] / self.denoising_steps)
                                  * self.dit_times[x.resolution][temp_max_gpus_num[x.request_id]])'''

        remove_groups = []
        update_groups = []
        for seq_group in temp_sorted_requests:
            if free_gpu_num < 1:
                continue
            #if seq_group not in self.hungry_requests:
            #    free_gpu_num -= temp_max_gpus_num[seq_group.request_id]
            #    continue
            cur_workers_num = len(self.requests_workers_ids[seq_group.request_id])
            if self.opt_gps_num[seq_group.request_id] == 4:
                if cur_workers_num + free_gpu_num >= 4:
                    count = 4 - cur_workers_num
                    j = 0
                    for id in cur_free_gpus_list:
                        if j >= count:
                            break
                        if self.gpu_status[id] == 0:
                            self.gpu_status[id] = 1
                            self.requests_workers_ids[seq_group.request_id].append(id)
                            seq_group.worker_ids.append(id)
                            j += 1
                    free_gpu_num -= count
                    remove_groups.append(seq_group.request_id)
                    update_groups.append(seq_group.request_id)
                    del self.requests_cur_steps[seq_group.request_id]
                    del self.requests_last_steps[seq_group.request_id]
                
                elif cur_workers_num + free_gpu_num == 2 or cur_workers_num + free_gpu_num == 3:
                    count = 2 - cur_workers_num
                    j = 0
                    for id in cur_free_gpus_list:
                        if j >= count:
                            break
                        if self.gpu_status[id] == 0:
                            self.gpu_status[id] = 1
                            self.requests_workers_ids[seq_group.request_id].append(id)
                            seq_group.worker_ids.append(id)
                            j += 1
                    free_gpu_num -= count
                    update_groups.append(seq_group.request_id)
                    self.requests_last_steps[seq_group.request_id] = self.requests_cur_steps[seq_group.request_id]
            
            elif self.opt_gps_num[seq_group.request_id] == 2:
                if cur_workers_num + free_gpu_num >= 2:
                    count = 2 - cur_workers_num
                    j = 0
                    for id in cur_free_gpus_list:
                        if j >= count:
                            break
                        if self.gpu_status[id] == 0:
                            self.gpu_status[id] = 1
                            self.requests_workers_ids[seq_group.request_id].append(id)
                            seq_group.worker_ids.append(id)
                            j += 1
                    free_gpu_num -= count
                    remove_groups.append(seq_group.request_id)
                    update_groups.append(seq_group.request_id)
                    del self.requests_cur_steps[seq_group.request_id]
                    del self.requests_last_steps[seq_group.request_id]

        if update_groups:
            print("update_groups ", len(update_groups))
        for group_id in update_groups:
            pload = {
                "request_id": group_id,
                "worker_ids": self.requests_workers_ids[group_id],
            }
            api_url = "http://127.0.0.1:8000/request_workers"
            response = self.post_http_request(pload, api_url)
        
        for group_id in remove_groups:
            del self.hungry_requests[group_id]
            del self.requests_workers_ids[group_id]

    def schedule(self) -> SequenceGroup:
        if self.waiting:
            seq_group = self.waiting[0]
            temp_worker_ids = [i for i in range(len(self.gpu_status)) if self.gpu_status[i] == 0]
            
            if seq_group.resolution == "360p":
                if len(temp_worker_ids) >= 4:
                    worker_ids = [temp_worker_ids[i] for i in range(4)]
                    for i in range(4):
                        self.gpu_status[temp_worker_ids[i]] = 1
                    seq_group.worker_ids = worker_ids
                    self.waiting.popleft()
                    return seq_group
                
                elif len(temp_worker_ids) == 2 or len(temp_worker_ids) == 3:
                    worker_ids = [temp_worker_ids[i] for i in range(2)]
                    for i in range(2):
                        self.gpu_status[temp_worker_ids[i]] = 1
                    seq_group.worker_ids = worker_ids
                    self.hungry_requests[seq_group.request_id] = seq_group
                    self.requests_workers_ids[seq_group.request_id] = worker_ids
                    self.requests_cur_steps[seq_group.request_id] = 0
                    self.requests_last_steps[seq_group.request_id] = 0
                    self.waiting.popleft()
                    return seq_group
                
                elif len(temp_worker_ids) == 1:
                    worker_ids = [temp_worker_ids[0]]
                    self.gpu_status[temp_worker_ids[0]] = 1
                    seq_group.worker_ids = worker_ids
                    self.hungry_requests[seq_group.request_id] = seq_group
                    self.requests_workers_ids[seq_group.request_id] = worker_ids
                    self.requests_cur_steps[seq_group.request_id] = 0
                    self.requests_last_steps[seq_group.request_id] = 0
                    self.waiting.popleft()
                    return seq_group
            
            elif seq_group.resolution == "240p":
                if len(temp_worker_ids) >= 2:
                    worker_ids = [temp_worker_ids[i] for i in range(2)]
                    for i in range(2):
                        self.gpu_status[temp_worker_ids[i]] = 1
                    seq_group.worker_ids = worker_ids
                    self.waiting.popleft()
                    return seq_group
                
                elif len(temp_worker_ids) == 1:
                    worker_ids = [temp_worker_ids[0]]
                    self.gpu_status[temp_worker_ids[0]] = 1
                    seq_group.worker_ids = worker_ids
                    self.hungry_requests[seq_group.request_id] = seq_group
                    self.requests_workers_ids[seq_group.request_id] = worker_ids
                    self.requests_cur_steps[seq_group.request_id] = 0
                    self.requests_last_steps[seq_group.request_id] = 0
                    self.waiting.popleft()
                    return seq_group
            
            elif seq_group.resolution == "144p":
                if len(temp_worker_ids) >= 1:
                    worker_ids = [temp_worker_ids[0]]
                    self.gpu_status[temp_worker_ids[0]] = 1
                    seq_group.worker_ids = worker_ids
                    self.waiting.popleft()
                    return seq_group
        
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