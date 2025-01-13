from typing import Dict, Deque, List

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