from typing import Deque, Dict, Tuple, List, Optional
from collections import deque
from videosys.core.sequence import SequenceGroup
import time

# add cost model to estimate the execution time in advance
class CostModel:
    def __init__(self):
        pass
    
    # find the min number of gpus to attain the slo and return the estimated execution time
    def get_min_gpus_num(self, 
                         model_type: str,
                         num_sampling_steps: int,
                         cfg_scale: float,
                         resolution: str,
                         aspect_ratio: str,
                         num_frames: int,
                         wait_time: Optional[float] = None,
                         slo_time: Optional[float] = None) -> Tuple[int, float]:
        pass

    def get_time(self,
                 gpus_num: int,
                 model_type: str,
                 num_sampling_steps: int,
                 cfg_scale: float,
                 resolution: str,
                 aspect_ratio: str,
                 num_frames: int) -> float:
        pass

class VideoScheduler:
    def __init__(
        self,
        enable_min_cost: Optional[bool] = False,
        enable_slo: Optional[bool] = False,
        slo_times: Optional[Dict[str, int]] = None
    ) -> None:
        # Sequence groups in the WAITING state.
        self.waiting: Deque[SequenceGroup] = deque()
        self.num_gpus = 2
        self.gpu_status = [0,0]
        self.num = 0
        self.cost_model = CostModel()
        self.enable_min_cost = enable_min_cost
        self.enable_slo = enable_slo
        self.slo_times = slo_times
        if self.slo_times:
            self.gpu_free_time = [0.0 for _ in range(self.num_gpus)]
    
    def schedule(self) -> Optional[SequenceGroup]:
        # sche in fcfs & bs == 1
        temp_time = time.time()
        if self.waiting:
            seq_group = self.waiting[0]
            # start scheduling by policies
            if self.enable_slo:
                # add the default settings
                cur_min_gpus_num, cur_min_time = self.cost_model.get_min_gpus_num(model_type = "OpenSora",
                                                                num_sampling_steps = 30,
                                                                cfg_scale = 7.0,
                                                                resolution = seq_group.resolution,
                                                                aspect_ratio = seq_group.aspect_ratio,
                                                                num_frames = seq_group.num_frames,
                                                                wait_time = temp_time - seq_group.add_time,
                                                                slo_time = self.slo_times[seq_group.resolution])
                cur_free_gpus_num = sum(1 for status in self.gpu_status if status == 0)
                if cur_min_gpus_num <= cur_free_gpus_num:
                    cur_worker_ids = []
                    for id, status in enumerate(self.gpu_status):
                        if status == 0:
                            cur_worker_ids.append(id)
                            self.gpu_status[id] = 1
                            self.gpu_free_time[id] = temp_time + cur_min_time
                            if len(cur_worker_ids) == cur_min_gpus_num:
                                break
                    seq_group.worker_ids = cur_worker_ids
                    self.waiting.popleft()
                    return seq_group
                else:
                    cur_used_gpus_free_time = [free_time for id, free_time in enumerate(self.gpu_free_time) if self.gpu_status[id] == 1]
                    # can be optimized -> find the Kth min number
                    cur_used_gpus_free_time.sort(key = lambda x: x)
                    # upper bound
                    '''expected_min_exe_time = self.cost_model.get_time(gpus_num = self.num_gpus,
                                                model_type = "OpenSora",
                                                num_sampling_steps = 30,
                                                cfg_scale = 7.0,
                                                resolution = seq_group.resolution,
                                                aspect_ratio = seq_group.aspect_ratio,
                                                num_frames = seq_group.num_frames)
                    if expected_min_exe_time + cur_used_gpus_free_time[-1] - seq_group.add_time <= self.slo_times[seq_group.resolution]:
                        return None'''
                    for i, start_time in enumerate(cur_used_gpus_free_time[cur_min_gpus_num - cur_free_gpus_num - 1: -1]):
                        cur_expected_exe_time = self.cost_model.get_time(gpus_num =  cur_min_gpus_num + i,
                                                                         model_type = "OpenSora",
                                                                         num_sampling_steps = 30,
                                                                         cfg_scale = 7.0,
                                                                         resolution = seq_group.resolution,
                                                                         aspect_ratio = seq_group.aspect_ratio,
                                                                         num_frames = seq_group.num_frames)
                        if cur_expected_exe_time + start_time - seq_group.add_time <= self.slo_times[seq_group.resolution]:
                            return None
                    # blocked already -> give all free gpus to retrieve
                    cur_worker_ids = [id for id, status in enumerate(self.gpu_status) if status == 0]
                    self.gpu_status = [1 for _ in range(self.num_gpus)]
                    cur_min_expected_exe_time = self.cost_model(gpus_num =  cur_free_gpus_num,
                                                                model_type = "OpenSora",
                                                                num_sampling_steps = 30,
                                                                cfg_scale = 7.0,
                                                                resolution = seq_group.resolution,
                                                                aspect_ratio = seq_group.aspect_ratio,
                                                                num_frames = seq_group.num_frames)
                    for id in cur_worker_ids:
                        self.gpu_free_time[id] = temp_time + cur_min_expected_exe_time
                        seq_group.worker_ids = cur_worker_ids
                        self.waiting.popleft()
                        return seq_group
            elif self.enable_min_cost:
                cur_min_gpus_num, _ = self.cost_model.get_min_gpus_num(model_type = "OpenSora",
                                                                num_sampling_steps = 30,
                                                                cfg_scale = 7.0,
                                                                resolution = seq_group.resolution,
                                                                aspect_ratio = seq_group.aspect_ratio,
                                                                num_frames = seq_group.num_frames)
                cur_free_gpus_num = sum(1 for status in self.gpu_status if status == 0)
                if cur_min_gpus_num <= cur_free_gpus_num:
                    cur_worker_ids = []
                    for id, status in enumerate(self.gpu_status):
                        if status == 0:
                            cur_worker_ids.append(id)
                            self.gpu_status[id] = 1
                            if len(cur_worker_ids) == cur_min_gpus_num:
                                break
                    seq_group.worker_ids = cur_worker_ids
                    self.waiting.popleft()
                    return seq_group
                # add down-degreed process
                else:
                    return None
            else:
                worker_ids = [0, 1]
                self.gpu_status = [1,1]
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