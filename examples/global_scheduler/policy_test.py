from typing import Dict, List, Tuple, Optional, Union, Deque
import threading
import time
import argparse
import random
from collections import deque

class Request:
     def __init__(self,
                  id: int,
                  resolution: str,
                  add_time: float) -> None:
                  self.id = id
                  self.resolution = resolution
                  self.add_time = add_time

class Multi_GPU_Type_Resources_Pool:
     def __init__(self,
                  log_file_path: str,
                  type1_num: int,
                  type2_num: int,
                  type4_num: int,
                  type1_slo: Optional[float] = None,
                  type2_slo: Optional[float] = None,
                  type4_slo: Optional[float] = None) -> None:
                  self.log_file_path = log_file_path
                  self.gpu_types_num: Dict[int, int] = {1: type1_num, 2: type2_num, 4: type4_num}
                  self.gpu_status: List[int] = [0 for _ in range(type1_num + type2_num * 2 + type4_num * 4)]
                  self.gpu_free_time: List[float] = [0.0 for _ in range(type1_num + type2_num * 2 + type4_num * 4)]
                  self.dit_time_configs: Dict[str, Dict[int, float]] = {"144p": {1: 3, 2: 3.4, 4: 3.5}, 
                                                                      "240p": {1: 8.3, 2: 4.6, 4: 3.7}, 
                                                                      "360p": {1: 19.2, 2: 10.4, 4: 6.1}}
                  self.vae_time_configs: Dict[str, Dict[int, float]] = {"144p": {1: 0.16, 2: 0.16, 4: 0.16}, 
                                                                      "240p": {1: 0.38, 2: 0.38, 4: 0.38}, 
                                                                      "360p": {1: 0.87, 2: 0.87, 4: 0.87}}
                  self.opt_gpu_config: Dict[str, int] = {"144p": 1, "240p": 2, "360p": 4}
                  self.slo_times: Dict[str, float] = {"144p": type1_slo, "240p": type2_slo, "360p": type4_slo}
                  self.type1_lock = threading.Lock()
                  self.type2_lock = threading.Lock()
                  self.type4_lock = threading.Lock()
                  self.all_type_lock = threading.Lock()  
                  self.gpu_status_lock = threading.Lock()   
                  self.logs_lock = threading.Lock()
     
     def get_min_gpu_num(self,
                         waiting_time: float,
                         slo_time: float,
                         resolution: str) -> Tuple[int, float]:
               pass
     
     def get_execution_time(self,
                            gpu_num: int,
                            resolution: str) -> float:
               pass
     
     def write_logs(self,
                    request: Request,
                    end_time: float) -> None:
               with self.logs_lock:
                    with open(self.log_file_path, 'a') as file:
                         file.write(f"request {request.id} ends at {end_time} with resolution {request.resolution}\n")

     def require_gpu_resources(self, 
                               request_type: str,
                               add_time: float,    
                               cluster_isolated: Optional[bool] = True,
                               round_robin: Optional[bool] = True,
                               round_robin_gpu_num: Optional[int] = 1,
                               best_match: Optional[bool] = True) -> Union[Tuple[bool, int, float], Tuple[bool, List[int], float]]:
               if cluster_isolated:
                    opt_gpu_num = self.opt_gpu_config[request_type]
                    if opt_gpu_num == 1:
                         with self.type1_lock:
                              if self.gpu_types_num[opt_gpu_num] > 0:
                                   self.gpu_types_num[opt_gpu_num] -= 1
                                   return (True, opt_gpu_num, self.dit_time_configs[request_type][opt_gpu_num] + 
                                   self.vae_time_configs[request_type][opt_gpu_num])
                              else:
                                   return (False, -1, -1)
                    elif opt_gpu_num == 2:
                         with self.type2_lock:
                              if self.gpu_types_num[opt_gpu_num] > 0:
                                   self.gpu_types_num[opt_gpu_num] -= 1
                                   return (True, opt_gpu_num, self.dit_time_configs[request_type][opt_gpu_num] + 
                                   self.vae_time_configs[request_type][opt_gpu_num])
                              else:
                                   return (False, -1, -1)
                    else:
                         with self.type4_lock:
                              if self.gpu_types_num[opt_gpu_num] > 0:
                                   self.gpu_types_num[opt_gpu_num] -= 1
                                   return (True, opt_gpu_num, self.dit_time_configs[request_type][opt_gpu_num] + 
                                   self.vae_time_configs[request_type][opt_gpu_num])
                              else:
                                   return (False, -1, -1)
               elif round_robin:
                    with self.all_type_lock:
                         if self.gpu_types_num[round_robin_gpu_num] > 0:
                              self.gpu_types_num[round_robin_gpu_num] -= 1
                              return (True, round_robin_gpu_num, self.dit_time_configs[request_type][round_robin_gpu_num] + 
                              self.vae_time_configs[request_type][round_robin_gpu_num])
                         else:
                              return (False, -1, -1)
               elif best_match:
                    with self.all_type_lock:
                         opt_gpu_num = self.opt_gpu_config[request_type]
                         if opt_gpu_num == 1:
                              if self.gpu_types_num[opt_gpu_num] > 0:
                                   self.gpu_types_num[opt_gpu_num] -= 1
                                   return (True, opt_gpu_num, self.dit_time_configs[request_type][opt_gpu_num] + 
                                   self.vae_time_configs[request_type][opt_gpu_num])
                              else:
                                   return (False, -1, -1)
                         elif opt_gpu_num == 2:
                              if self.gpu_types_num[opt_gpu_num] > 0:
                                   self.gpu_types_num[opt_gpu_num] -= 1
                                   return (True, opt_gpu_num, self.dit_time_configs[request_type][opt_gpu_num] + 
                                   self.vae_time_configs[request_type][opt_gpu_num])
                              elif self.gpu_types_num[1] > 0:
                                   self.gpu_types_num[1] -= 1
                                   return (True, 1, self.dit_time_configs[request_type][1] + 
                                   self.vae_time_configs[request_type][1])
                              else:
                                   return (False, -1, -1)
                         else:
                              if self.gpu_types_num[opt_gpu_num] > 0:
                                   self.gpu_types_num[opt_gpu_num] -= 1
                                   return (True, opt_gpu_num, self.dit_time_configs[request_type][opt_gpu_num] + 
                                   self.vae_time_configs[request_type][opt_gpu_num])
                              elif self.gpu_types_num[2] > 0:
                                   self.gpu_types_num[2]  -= 1
                                   return (True, 2, self.dit_time_configs[request_type][2] + 
                                   self.vae_time_configs[request_type][2])
                              elif self.gpu_types_num[1] > 0:
                                   self.gpu_types_num[1] -= 1
                                   return (True, 1, self.dit_time_configs[request_type][1] + 
                                   self.vae_time_configs[request_type][1])
                              else:
                                   return (False, -1, -1)
               else:
                    with self.gpu_status_lock:
                         cur_time = time.time()
                         cur_min_gpu_num, cur_expected_exe_time = self.get_min_gpu_num(waiting_time = cur_time - add_time,
                                                                                  slo_time = self.slo_times[request_type],
                                                                                  resolution = request_type)
                         cur_free_gpu_ids = [idx for idx, status in enumerate(self.gpu_status) if status == 0]
                         cur_free_gpu_num = len(cur_free_gpu_ids)
                         cur_allocate_gpu_ids: List[int] = []
                         if cur_min_gpu_num <= cur_free_gpu_num:
                              cur_allocate_gpu = 0
                              for idx in cur_free_gpu_ids:
                                   self.gpu_status[idx] = 1
                                   self.gpu_free_time[idx] = cur_time + cur_expected_exe_time
                                   cur_allocate_gpu_ids.append(idx)
                                   cur_allocate_gpu += 1
                                   if cur_allocate_gpu == cur_min_gpu_num:
                                        break
                              return (True, cur_allocate_gpu_ids, cur_expected_exe_time)
                         else:
                              cur_used_gpu_free_time = [free_time for idx, free_time in enumerate(self.gpu_free_time) if self.gpu_status[idx] == 1]
                              cur_used_gpu_free_time.sort(key = lambda x: x)
                              for i, start_time in enumerate(cur_used_gpu_free_time[cur_min_gpu_num - cur_free_gpu_num - 1: -1]):
                                   expected_exe_time = self.get_execution_time(gpu_num = cur_min_gpu_num + i,
                                                                               resolution = request_type)
                                   if expected_exe_time + start_time - add_time <= self.slo_times[request_type]:
                                        return (False, -1, -1)
                              cur_min_expected_exe_time = self.get_execution_time(gpu_num = cur_free_gpu_num,
                                                                                  resolution = request_type)
                              for idx in cur_free_gpu_ids:
                                   self.gpu_status[idx] = 1
                                   self.gpu_free_time[idx] = cur_time + cur_min_expected_exe_time
                              return (True, cur_free_gpu_ids, cur_min_expected_exe_time)
     
     def release_gpu_resources(self, 
                               release_gpu_num: int, 
                               cluster_isolated: Optional[bool] = True,
                               slo_required: Optional[bool] = True,
                               allocated_gpu_ids: Optional[List[int]] = None) -> None:
               if cluster_isolated:
                    if release_gpu_num == 1:
                         with self.type1_lock:
                              self.gpu_types_num[release_gpu_num] += 1
                    elif release_gpu_num == 2:
                         with self.type2_lock:
                              self.gpu_types_num[release_gpu_num] += 1
                    else:
                         with self.type4_lock:
                              self.gpu_types_num[release_gpu_num] += 1
               elif slo_required:
                    with self.gpu_status_lock:
                         for idx in allocated_gpu_ids:
                              self.gpu_status[idx] = 0
               else:
                    with self.all_type_lock:
                         self.gpu_types_num[release_gpu_num] += 1
     
def thread_function(request: Request,
                    gpu_resources_pool: Multi_GPU_Type_Resources_Pool,
                    release_gpu_num: int, 
                    cluster_isolated: Optional[bool] = True,
                    slo_required: Optional[bool] = True,
                    allocated_gpu_ids: Optional[List[int]] = None,
                    expected_exe_time: Optional[float] = None) -> None:
               time.sleep(expected_exe_time)
               gpu_resources_pool.release_gpu_resources(release_gpu_num = release_gpu_num,
                                                        cluster_isolated = cluster_isolated,
                                                        slo_required = slo_required,
                                                        allocated_gpu_ids = allocated_gpu_ids)
               end_time = time.time()
               gpu_resources_pool.write_logs(request = request,
                                             end_time = end_time)

def fcfs_scheduler(gpu_resources_pool: Multi_GPU_Type_Resources_Pool, 
                   thread_dequeue: Deque[Request],
                   cluster_isolated: Optional[bool] = True,
                   round_robin: Optional[bool] = True,
                   best_match: Optional[bool] = True,
                   slo_required: Optional[bool] = True) -> None:
               activate_threads: List[threading.Thread] = []
               count = 0
               gpu_num = [1, 2, 4]
               while thread_dequeue:
                    if round_robin:
                         cur_require_gpu = gpu_num[count % 3]
                         cur_request = thread_dequeue.popleft()
                         can_exe, allocate_gpu_num, expected_exe_time = gpu_resources_pool.require_gpu_resources(request_type = cur_request.resolution,
                                                                                                                 add_time = cur_request.add_time,
                                                                                                                 cluster_isolated = cluster_isolated,
                                                                                                                 round_robin = round_robin,
                                                                                                                 round_robin_gpu_num = cur_require_gpu,
                                                                                                                 best_match = best_match)
                         if can_exe:
                              cur_thread = threading.Thread(target = thread_function, args = (cur_request,
                                                                                               gpu_resources_pool,
                                                                                               allocate_gpu_num,
                                                                                               cluster_isolated, 
                                                                                               slo_required, 
                                                                                               None, 
                                                                                               expected_exe_time))
                              cur_thread.start()
                              activate_threads.append(cur_thread)
                         else:
                              thread_dequeue.append(cur_request)
                         count += 1
                    elif slo_required:
                         cur_request = thread_dequeue.popleft()
                         can_exe, allocate_gpu_num_ids, expected_exe_time = gpu_resources_pool.require_gpu_resources(request_type = cur_request.resolution,
                                                                                                                     add_time = cur_request.add_time,
                                                                                                                     cluster_isolated = cluster_isolated,
                                                                                                                     round_robin = round_robin,
                                                                                                                     round_robin_gpu_num = -1,
                                                                                                                     best_match = best_match)
                         if can_exe:
                              cur_thread = threading.Thread(target = thread_function, args = (cur_request,
                                                                                               gpu_resources_pool, 
                                                                                               -1,
                                                                                               cluster_isolated, 
                                                                                               slo_required, 
                                                                                               allocate_gpu_num_ids, 
                                                                                               expected_exe_time))
                              cur_thread.start()
                              activate_threads.append(cur_thread)
                         else:
                              thread_dequeue.append(cur_request)
                    else:
                         cur_request = thread_dequeue.popleft()
                         can_exe, allocate_gpu_num, expected_exe_time = gpu_resources_pool.require_gpu_resources(request_type = cur_request.resolution,
                                                                                                                 add_time = cur_request.add_time,
                                                                                                                 cluster_isolated = cluster_isolated,
                                                                                                                 round_robin = round_robin,
                                                                                                                 round_robin_gpu_num = -1,
                                                                                                                 best_match = best_match)
                         if can_exe:
                              cur_thread = threading.Thread(target = thread_function, args = (cur_request,
                                                                                               gpu_resources_pool,
                                                                                               allocate_gpu_num,
                                                                                               cluster_isolated, 
                                                                                               slo_required, 
                                                                                               None, 
                                                                                               expected_exe_time))
                              cur_thread.start()
                              activate_threads.append(cur_thread)
                         else:
                              thread_dequeue.append(cur_request)
               for thread in activate_threads:
                    thread.join()

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--log", type = str, default = "/home/jovyan/hhy/VideoSys/examples/global_scheduler")
     parser.add_argument("--cluster-isolated", action = "store_true", default = False)
     parser.add_argument("--round-robin", action = "store_true", default = False)
     parser.add_argument("--best-match", action = "store_true", default = False)
     parser.add_argument("--slo-required", action = "store_true", default = False)
     parser.add_argument("--request-num", type = int, default = 20)
     parser.add_argument("--type1-num", type = int, default = 6)
     parser.add_argument("--type2-num", type = int, default = 3)
     parser.add_argument("--type4-num", type = int, default = 1)
     parser.add_argument("--type1-slo", type = float, default = -1.0)
     parser.add_argument("--type2-slo", type = float, default = -1.0)
     parser.add_argument("--type4-slo", type = float, default = -1.0)
     args = parser.parse_args()

     resolutions = ["144p", "240p", "360p"]
     resolutions_weights = [args.type1_num, args.type2_num, args.type4_num]

     gpu_resources_pool = Multi_GPU_Type_Resources_Pool(log_file_path = args.log,
                                                        type1_num = args.type1_num, 
                                                        type2_num = args.type2_num, 
                                                        type4_num = args.type4_num,
                                                        type1_slo = args.type1_slo,
                                                        type2_slo = args.type2_slo,
                                                        type4_slo = args.type4_slo)
     
     requests: Deque[Request] = deque()
     add_time = time.time()
     for i in range(args.request_num):
          resolution = random.choices(resolutions, [args.type1_num, args.type2_num, args.type4_num], k = 1)[0]
          requests.append(Request(id = i, resolution = resolution, add_time = add_time))
     
     print(f"test starts at {add_time}")
     
     fcfs_scheduler(gpu_resources_pool = gpu_resources_pool, 
                    thread_dequeue = requests,
                    cluster_isolated = args.cluster_isolated,
                    round_robin = args.round_robin,
                    best_match = args.best_match,
                    slo_required = args.slo_required)