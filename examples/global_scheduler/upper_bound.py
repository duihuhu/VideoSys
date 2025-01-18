from typing import List, Optional, Dict
import math
import argparse

dit_times: Dict[int, Dict[int, float]] = {0: {1: 3, 2: 3.4, 4: 3.5}, 
                                          1: {1: 8.3, 2: 4.6, 4: 3.7}, 
                                          2: {1: 19.2, 2: 10.4, 4: 6.1}}
vae_times: Dict[int, Dict[int, float]] = {0: {1: 0.16, 2: 0.16, 4: 0.16}, 
                                          1: {1: 0.38, 2: 0.38, 4: 0.38}, 
                                          2: {1: 0.87, 2: 0.87, 4: 0.87}}

def try_best_allocate(st: int, ed: int, gpus_per_instance: int, process_group_size: int) -> int:
    st_row = st // gpus_per_instance
    st_column = st % gpus_per_instance
    ed_row = ed // gpus_per_instance
    ed_column = ed % gpus_per_instance
    ans = 0
    if st_row != ed_row:
        ans += ((gpus_per_instance - st_column) // process_group_size)
        ans += ((ed_column + 1) // process_group_size)
        mid_row_num = ed_row - st_row -1
        if mid_row_num >= 1:
            ans += (gpus_per_instance // process_group_size)
    else:
        ans += ((ed_column - st_column + 1) // process_group_size)
    return ans

def upper_bound_solver(batch: bool, 
                       requests_num: int,
                       weights: List[int],
                       instances_num: int, 
                       gpus_per_instance: int,
                       arrival_ratio: Optional[float]) -> float:
    gpus_num = instances_num * gpus_per_instance
    types_num = len(weights)
    total_weights = sum(weights)
    tasks_per_type: List[int] = []
    for weight in weights:
        tasks_per_type.append(round((weight / total_weights) * requests_num))
    dp = [[float('inf') for _ in range(types_num + 1)] for _ in range(gpus_num + 1)]
    for i in range(1, gpus_num + 1):
        dp[i][0] = 0.0
    for j in range(1, types_num + 1):
        dp[0][j] = float('inf')
  
    for i in range(1, gpus_num + 1):
        for j in range(1, types_num + 1):
            for k in range(1, i):
                for p in [1, 2, 4]:
                    model_replicas = try_best_allocate(i - k, i - 1, gpus_per_instance, p)
                    #print(f"model_replicas {model_replicas}")
                    if model_replicas == 0:
                        continue
                    iteration_time = dit_times[j - 1][p] + vae_times[j - 1][p]
                    #print(f"ite time {iteration_time}")
                    if batch:
                        iterations_num = tasks_per_type[j - 1] // model_replicas + 1
                        cumulative_resource_occupancy_time = iteration_time * iterations_num
                        #print(f"i-k {i-k} j-1 {j-1} {k * cumulative_resource_occupancy_time}")
                        dp[i][j] = min(dp[i][j], dp[i - k][j - 1] + k * cumulative_resource_occupancy_time)
                    else:
                        task_ratio = weights[j - 1] / total_weights
                        utilization_ratio = (arrival_ratio * task_ratio * iteration_time) / model_replicas
                        r = arrival_ratio * task_ratio * iteration_time
                        ra = r ** model_replicas
                        af = math.factorial(model_replicas)
                        ps = 0.0
                        for s in range(0, model_replicas):
                            ps += (r ** s) / math.factorial(s)
                        p0 = 1 / (ra / (af * (1 - utilization_ratio)) + ps)
                        cumulative_resource_occupancy_time = (iteration_time + ((ra * iteration_time) / (af * model_replicas * (1 - utilization_ratio) ** 2)) * p0) / 2
                        dp[i][j] = min(dp[i][j], dp[i - k][j - 1] + k * cumulative_resource_occupancy_time)
            #print(f"dp[{i}][{j}] {dp[i][j]}")
    
    return dp[gpus_num][types_num]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type = int, default = 8)
    parser.add_argument("--gpus", type = int, default = 8)
    parser.add_argument("--weight1", type = int, default = 1)
    parser.add_argument("--weight2", type = int, default = 1)
    parser.add_argument("--weight3", type = int, default = 1)
    parser.add_argument("--num", type = int, default = 128)
    parser.add_argument("--lam", type = float, default = 4.0)
    parser.add_argument("--batch", action = "store_true", default = True)
    args = parser.parse_args()

    weigths: List[int] = [args.weight1, args.weight2, args.weight3]
    output = upper_bound_solver(args.batch, args.num, weigths, args.instances, args.gpus, args.lam)
    print(output)