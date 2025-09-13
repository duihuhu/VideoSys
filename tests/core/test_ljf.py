import heapq
from typing import Callable, List, Optional, Tuple, Dict
import uuid

denoising_steps: int = 30
res_to_dit_times: Dict[str, Dict[int, float]] = {"144p": {1: 4.45, 2: 3.67, 4: 3.69, 8: 3.98}, 
                                                 "240p": {1: 11.27, 2: 5.88, 4: 3.67, 8: 4.10}, 
                                                 "360p": {1: 25.05, 2: 11.91, 4: 6.68, 8: 4.16},
                                                 "480p": {1: 44.24, 2: 20.43, 4: 11.12, 8: 6.31},
                                                 "720p": {1: 112.76, 2: 48.64, 4: 25.08, 8: 13.39}}
res_to_vae_times: Dict[str, float] = {"144p": 0.34, "240p": 0.78, "360p": 1.81, "480p": 3.54, "720p": 8.70}
#res_to_opt_gpu_nums: Dict[str, int] = {"144p": 1, "240p": 2, "360p": 4, "480p": 8, "720p": 8}

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
    
    def __lt__(self, other):
        return self.request_id < other.request_id
    
    def cost(self, k: int) -> Optional[float]:
        """
        计算该请求分配 k 个 GPU 时的单资源时间 Ci
        - 若 k == 0 则 cost(0)=0
        - 若 k > max_k 则不可用，返回 None
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
        total_time = k * remaining_steps * dit_time + vae_time
        return total_time


def allocate_unit_skip(
    func: Callable[[Request, int], Optional[float]],
    N: List[Request],
    M: int,
    allow_jump: bool = True,
    verbose: bool = False
) -> Tuple[List[int], float, int]:
    """
    单位分配贪心（每次尝试 +1；若 func(i, k) 不接受该 k，则跳过该元素）
    - func(i, k): 当元素 i 被分到 k 个资源时，返回单资源占用时间 Ci (float)。
                    若该 (i, k) 不可用/不接受，func 应返回 None 或抛异常。
                    注意：不要假定 func(i, 0) 可用——本实现把 cost(0)=0 明确定义为0。
    - N: 元素个数（索引 0..N-1）
    - M: 总资源数（非负整数）
    - allow_jump: 若 True，则在 k+1 不可用时尝试向前搜索第一个可用的更大 k（即“跳过不可用 k”并做一次跳跃）。
                  这等价于把跳跃当作 chunk（不是你要求的纯 +1 策略），默认 False。
    - verbose: 若 True，会打印运行中提示（用于调试）
    返回: (Ki_list, total_cost, remaining)
        Ki_list: 长度 N 的整数列表，表示每个元素分到的资源数（可能无法消耗所有 M）
        total_cost: 累计总代价（等于 sum_i Ki[i] * func(i, Ki[i]) 对可用 Ki）
        remaining: 剩余未分配的资源数（若为0则刚好消耗 M）
    注意:
      - 该算法在 func 满足边际非减（在允许的 k 上）的情形下是全局最优的。
      - 当 allow_jump==False 如果 func(i, k+1) 不可用，则该元素不会再获得更多资源。
    """
    # 初始状态
    Ki: Dict[str, int] = {req.request_id: 0 for req in N}
    total_cost = 0.0
    remaining = M

    # helper: cost for k (with cost(0)=0)
    def cost_of(req: Request, k: int) -> float:
        if k <= 0:
            return 0.0
        v = func(req, k)
        if v is None:
            raise ValueError(f"func returned None for (req={req}, k={k}) when cost requested.")
        return float(v)

    # helper: find next k > cur_k such that func(i,k) is accepted (<= M)
    def next_accepted_k(req: Request, cur_k: int) -> Optional[int]:
        # linear scan forward until M
        k = cur_k + 1
        while k <= M:
            try:
                v = func(req, k)
            except Exception:
                v = None
            if v is not None:
                return k
            k += 1
        return None

    # 初始化堆：只把那些能接受 k=1 的元素入堆（表示可做一次 +1）
    # 堆元素：(marginal_cost, i, cur_k, next_k)
    # - cur_k 是当前该元素的 k（初始 0）
    # - next_k 是用于计算这个 marginal 的目标 k（通常 cur_k+1 或更大的跳跃 if allow_jump）
    heap: List[Tuple[float, Request, int, int]] = []
    for req in N:
        # try to see if we can give first unit
        try:
            k1 = next_accepted_k(req, 0) if allow_jump else (1 if func(req, 1) is not None else None)
        except Exception:
            k1 = None
        if k1 is None:
            # 如果 k1 不可用（在 allow_jump=False 时表示 func(i,1) 不可用），就跳过此元素
            continue
        # 计算边际 cost: cost(k1) - cost(0)=cost(k1)
        try:
            delta = cost_of(req, k1) - 0.0
        except ValueError:
            continue
        # push (delta, i, cur_k=0, next_k=k1)
        heap.append((delta, req, 0, k1))
        print(f"Init heap push: req={req.request_id} ({req.res}), delta={delta:.6g}, next_k={k1}")
    heapq.heapify(heap)

    # 分配循环：每次从堆里 pop 最小边际，接受该次“（可能跳跃的）+1”分配
    # 注意：这里把 next_k 视作一次“要把 cur_k 直接提高到 next_k”的操作；当 allow_jump=False 时 next_k == cur_k+1
    while remaining > 0 and heap:
        delta, req, cur_k, next_k = heapq.heappop(heap)
        # chunk_size = how many resources this operation consumes
        chunk_size = next_k - cur_k
        if chunk_size <= 0:
            # 理论上不会发生
            continue
        if chunk_size > remaining:
            # 剩余资源不够接受这个跳跃/块 —— 在 unit-only 模式允许跳过（即我们不拆分 chunk）
            # 我们选择跳过该项（不 push 回），继续尝试其他元素。
            # 如果 allow_jump == False 且 chunk_size == 1，这里不会走到这分支
            if verbose:
                print(f"Skip operation for req={req.request_id} ({req.res}) chunk_size={chunk_size} > remaining={remaining}")
            continue

        # 接受该操作：把 Ki[i] = next_k
        # 增加总成本为 cost(next_k) - cost(cur_k)
        try:
            cost_prev = cost_of(req, cur_k)
            cost_next = cost_of(req, next_k)
        except ValueError:
            # 如果突然发现 next_k 无效（并发变化/异常），跳过
            continue

        Ki[req.request_id] = next_k
        total_cost += (cost_next - cost_prev)
        remaining -= chunk_size
        if verbose:
            print(f"Assign element {req.request_id} ({req.res}): {cur_k} -> {next_k}, delta={delta:.6g}, remaining={remaining}")

        # 计算该元素的下一次候选（从 next_k 向前）
        if allow_jump:
            nk = next_accepted_k(req, next_k)
        else:
            # 仅尝试 next_k = next_k + 1 （即单位尝试模式）
            try:
                nk_candidate = next_k + 1
                # 检查是否 func(req, nk_candidate) 可用（若不可用则不 push）
                v = func(req, nk_candidate)
                nk = nk_candidate if v is not None else None
            except Exception:
                nk = None

        if nk is not None and nk <= M:
            # compute marginal for pushing back
            try:
                new_delta = cost_of(req, nk) - cost_of(req, next_k)
            except ValueError:
                new_delta = None
            if new_delta is not None:
                print(f"Push back to heap: req={req.request_id} ({req.res}), delta={new_delta:.6g}, next_k={nk}")
                heapq.heappush(heap, (new_delta, req, next_k, nk))
        # else: no further candidate for this element (it becomes inactive)

    # 返回：Ki 列表，总成本，剩余未分配资源
    if verbose and remaining > 0:
        print(f"Allocation ended with remaining={remaining} resources (heap empty or all candidates too big).")
    return Ki, total_cost, remaining


# -----------------------------
# 示例（可以直接运行）
# -----------------------------
if __name__ == "__main__":
    '''
    # 示例 func：元素 i 在某些 k 不接受（返回 None），其它返回 Ci
    def example_func(i, k):
        # 比如第 0 个元素在 k==2 不接受（返回 None），其它都接受
        if i == 0 and k == 2:
            return None
        # 随机造个 func（允许负边际）
        # 例如 func(i,k) = base[i] + slope[i] * k + special dip at some k
        base = [2.0, 1.0, 3.0]
        slope = [0.1, -0.05, 0.0]
        v = base[i] + slope[i] * k
        return v

    N = 3
    M = 5

    Ki, total, remaining = allocate_unit_skip(example_func, N, M, allow_jump=False, verbose=True)
    print("Result (no jump): Ki =", Ki, "total =", total, "remaining =", remaining)

    # 若要允许跳跃（跳过不可用 k=2 并直接到 k=3），可以:
    Ki2, total2, rem2 = allocate_unit_skip(example_func, N, M, allow_jump=True, verbose=True)
    print("Result (allow jump): Ki =", Ki2, "total =", total2, "remaining =", rem2)
    '''
    M = 8
    N = 8
    requests: List[Request] = [Request(res="144p", inflight=False, cur_steps=0, cur_gpus=0, request_id=random_uuid()),
                               Request(res="360p", inflight=False, cur_steps=0, cur_gpus=0, request_id=random_uuid()),
                               Request(res="720p", inflight=False, cur_steps=0, cur_gpus=0, request_id=random_uuid())]
                               #Request(res="360p", inflight=False, cur_steps=0, cur_gpus=0, request_id=random_uuid()),
                               #Request(res="360p", inflight=False, cur_steps=0, cur_gpus=0, request_id=random_uuid()),
                               #Request(res="720p", inflight=False, cur_steps=0, cur_gpus=0, request_id=random_uuid()),
                               #Request(res="360p", inflight=False, cur_steps=0, cur_gpus=0, request_id=random_uuid()),
                               #Request(res="360p", inflight=False, cur_steps=0, cur_gpus=0, request_id=random_uuid())]
    '''for _ in range(N):
        # 随机生成请求（res 任选，inflight 随机）
        import random
        random.seed(42)  # 为了可重复的随机结果
        res = random.choice(list(res_to_dit_times.keys()))
        req = Request(res = res, inflight = False, cur_steps = 0, cur_gpus = 0)
        requests.append(req)

    for req in requests:
        print(f"Request: res={req.res}, inflight={req.inflight}, cur_steps={req.cur_steps}, cur_gpus={req.cur_gpus}")
    '''
    for req in requests:
        print(f"Request ID: {req.request_id}, res={req.res}, inflight={req.inflight}, cur_steps={req.cur_steps}, cur_gpus={req.cur_gpus}")
    
    Ki, total, remaining = allocate_unit_skip(Request.cost, requests, M, allow_jump=True, verbose=True)
   
    print("Final allocation: ", "total cost =", total, "remaining =", remaining)
    for key, value in Ki.items():
        print(f"Request ID: {key}, Allocated GPUs: {value}")