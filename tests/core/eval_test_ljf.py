import math
from typing import Callable, List, Optional, Tuple, Dict
import uuid

denoising_steps: int = 30
res_to_dit_times: Dict[str, Dict[int, float]] = {"144p": {1: 4.45, 2: 3.67, 4: 3.69, 8: 3.98}, 
                                                 "240p": {1: 11.27, 2: 5.88, 4: 3.67, 8: 4.10}, 
                                                 "360p": {1: 25.05, 2: 11.91, 4: 6.68, 8: 4.16},
                                                 "480p": {1: 44.24, 2: 20.43, 4: 11.12, 8: 6.31},
                                                 "720p": {1: 112.76, 2: 48.64, 4: 25.08, 8: 13.39}}
res_to_vae_times: Dict[str, float] = {"144p": 0.34, "240p": 0.78, "360p": 1.81, "480p": 3.54, "720p": 8.70}

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

class Request:
    def __init__(self, res: str, 
                 inflight: Optional[bool] = False, 
                 cur_steps: Optional[int] = 0,
                 cur_gpus: Optional[int] = 0,
                 request_id: Optional[str] = None) -> None:
        self.res = res
        self.inflight = inflight
        self.cur_steps = cur_steps
        self.cur_gpus = cur_gpus
        self.request_id = request_id

    def cost(self, k: int) -> Optional[float]:
        """
        计算该请求分配 k 个 GPU 时的单资源时间 Ci
        - 若 k == 0 则 cost(0)=0
        - 若 k + cur_gpus_num 不为偶数则不可用，返回 None
        - 否则返回 Ci (float)
        """
        if k < 0 or ((k + self.cur_gpus) > 1 and ((k + self.cur_gpus) & 1)):
            return None
        if k == 0:
            return 0.0
        '''max_k = res_to_opt_gpu_nums[self.res]
        if k > max_k:
            return None'''
        # 计算剩余时间（秒）
        dit_time = res_to_dit_times[self.res][k]
        vae_time = res_to_vae_times[self.res]
        remaining_steps = max(denoising_steps - self.cur_steps, 0) if self.inflight else denoising_steps
        total_time = k * dit_time * (remaining_steps / denoising_steps) + vae_time
        return total_time

def enumerate_allocations(
    n: List[Request],
    M: int,
    func: Callable[[int, int], Optional[float]],
    max_k: Optional[int] = None,
    cost_is_total: bool = True,
    prune: bool = False,
    verbose: bool = True
) -> List[Tuple[Tuple[int, ...], float]]:
    """
    枚举所有分配方案 k_0..k_{n-1} 满足 sum(k_i) == M 的情况，并打印每个方案的总 cost。
    :param n: 元素数量
    :param M: 总资源数（要求 sum Ki = M）
    :param func: func(i, k) -> 若 cost_is_total==False 返回单资源时间 Ci（float），
                 则 cost_i(k) = k * Ci；若 cost_is_total==True 则 func 返回 cost_i(k) 的**总代价**。
                 若 (i,k) 不可用，func 可返回 None 或抛异常，那个 k 的方案会被跳过。
    :param max_k: 每个元素允许的最大 k（若 None 则默认为 M）
    :param cost_is_total: 如上说明
    :param prune: 若 True，会使用当前已知最小 cost 做简单剪枝（只影响速度，不影响最终打印的全量结果；
                  为了确保打印所有方案，建议默认 False）
    :param verbose: 是否打印每个方案（True）
    :return: 按 total_cost 升序排序的列表 [(alloc_tuple, total_cost), ...]
    """

    if max_k is None:
        max_k_list = [M] * len(n)
    else:
        max_k_list = [max_k] * len(n)

    results: List[Tuple[Tuple[int, ...], float]] = []
    best_cost = math.inf

    # DFS 生成所有满足总和为 M 的分配，current 为已分配的前缀，acc_cost 为已累积代价
    def dfs(idx: int, remaining: int, current: List[int], acc_cost: float):
        nonlocal best_cost

        # 若到达最后一个元素，直接确定它的 k = remaining（唯一选择）
        if idx == len(n) - 1:
            k = remaining
            if k < 0 or k > max_k_list[idx]:
                return
            # cost for this k
            if k == 0:
                cost_k = 0.0
            else:
                try:
                    v = func(n[idx], k)
                except Exception:
                    return
                if v is None:
                    return
                cost_k = float(v) if cost_is_total else float(k * v)
            total = acc_cost + cost_k
            alloc = tuple(current + [k])
            results.append((alloc, total))
            if verbose:
                print(f"分配方案: {alloc}, 总cost = {total}")
            if total < best_cost:
                best_cost = total
            return

        # 否则枚举当前元素可取的 k 从 0 到 min(max_k, remaining)
        limit = min(max_k_list[idx], remaining)
        for k in range(0, limit + 1):
            # cost for this k (k==0 cost==0，不调用 func)
            if k == 0:
                cost_k = 0.0
            else:
                try:
                    v = func(n[idx], k)
                except Exception:
                    # 认为这个 k 不可用
                    continue
                if v is None:
                    continue
                cost_k = float(v) if cost_is_total else float(k * v)

            new_acc = acc_cost + cost_k

            # 可选剪枝：如果开启 prune 并且 new_acc 已经 >= best_cost，则跳过分支
            if prune and new_acc >= best_cost:
                continue

            # 继续递归
            dfs(idx + 1, remaining - k, current + [k], new_acc)

    # 启动 DFS
    dfs(0, M, [], 0.0)

    # 按 cost 排序后返回
    results.sort(key=lambda x: x[1])
    return results


# -----------------------
# 示例用法（演示）
# -----------------------
if __name__ == "__main__":
    '''# 示例 func：返回每个元素的单资源时间 Ci（可能导致负边际）
    def example_func(i, k):
        # 下面只是举例：元素 0 在 k=1 很贵，在 k=2 很便宜（演示负 delta）
        if i == 0:
            if k == 1: return 10.0
            if k == 2: return 2.0
            if k == 3: return 3.0
            # 超出 3，我们设为不可用
            return None
        # 元素 1 相对稳定
        if i == 1:
            return 4.0 + 0.5 * k
        # 元素 2 接受任意 k，略微增长
        return 1.0 + 0.1 * k

    # 例：3 个元素，M = 3
    n = 3
    M = 3
    print("开始穷举所有分配（sum Ki = M）并打印：")
    all_results = enumerate_allocations(n, M, example_func, max_k=None, cost_is_total=False, prune=False, verbose=True)

    print("\n按总 cost 升序的前 5 个解（如果有）：")
    for alloc, cost in all_results[:5]:
        print(alloc, cost)'''
    M = 8
    N = 8
    requests: List[Request] = [Request(res="720p", inflight=False, cur_steps=0, cur_gpus=0, request_id=random_uuid()),
                               Request(res="720p", inflight=False, cur_steps=0, cur_gpus=0, request_id=random_uuid()),
                               Request(res="720p", inflight=False, cur_steps=0, cur_gpus=0, request_id=random_uuid()),
                               Request(res="720p", inflight=False, cur_steps=0, cur_gpus=0, request_id=random_uuid())]
    for req in requests:
        print(f"Request ID: {req.request_id}, res={req.res}, inflight={req.inflight}, cur_steps={req.cur_steps}, cur_gpus={req.cur_gpus}")
    results = enumerate_allocations(requests, M, Request.cost, max_k=None, cost_is_total=True, prune=False, verbose=False)
    print("前 5 个最优解（按总代价排序）：")
    for alloc, cost in results[:5]:
        print(alloc, cost)