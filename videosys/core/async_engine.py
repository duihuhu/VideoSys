from videosys import VideoSysEngine
import asyncio
import os
from typing import (Callable, Dict, List, Optional, Set, Tuple, Union)
from functools import partial
import time
from videosys.utils.logging import logger
from videosys.core.outputs import RequestOutput, KvPreparedResponse
from videosys import OpenSoraConfig
from videosys.utils.config import DeployConfig
# from videosys.pipelines.open_sora.video_ops import trans_ops
from videosys.core.engine import VideoSched
import threading
import aiohttp
from videosys.core.sequence import SequenceGroup
import queue
import requests
import copy
from queue import Queue
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

ENGINE_ITERATION_TIMEOUT_S = int(
    os.environ.get("VLLM_ENGINE_ITERATION_TIMEOUT_S", "60"))

class CommData:
    def __init__(self, headers, payload) -> None:
        self.headers = headers
        self.payload = payload

class AsyncEngineDeadError(RuntimeError):
    pass

def _raise_exception_on_finish(
        task: asyncio.Task, error_callback: Callable[[Exception],
                                                     None]) -> None:
    msg = ("Task finished unexpectedly. This should never happen! "
           "Please open an issue on Github.")

    exception = None
    try:
        task.result()
        # NOTE: This will be thrown if task exits normally (which it should not)
        raise AsyncEngineDeadError(msg)
    except Exception as e:
        exception = e
        logger.error("Engine background task failed", exc_info=e)
        error_callback(exception)
        raise AsyncEngineDeadError(
            msg + " See stack trace above for the actual cause.") from e

class AsyncStream:
    """A stream of RequestOutputs for a request that can be
    iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue = asyncio.Queue()
        self._finished = False

    def put(self, item: Union[RequestOutput, Exception]) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopAsyncIteration())
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        result = await self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result

class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream,
                                                dict]] = asyncio.Queue()
        self.new_requests_event = asyncio.Event()
        self._kv_responses: asyncio.Queue[Tuple[AsyncStream, dict]] = asyncio.Queue()
        
    def __contains__(self, item):
        return item in self._request_streams

    def __len__(self) -> int:
        return len(self._request_streams)

    def propagate_exception(self,
                            exc: Exception,
                            request_id: Optional[str] = None) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self._request_streams[request_id].put(exc)
            self.abort_request(request_id)
        else:
            for rid, stream in self._request_streams.items():
                stream.put(exc)
                self.abort_request(rid)

    def process_request_output(self,
                                global_ranks: List[int],
                               request_output: RequestOutput,
                               *,
                               verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id
        request_output.global_ranks = global_ranks
        self._request_streams[request_id].put(request_output)
        if request_output.finished:
            logger.info(f"Finished request {request_id}.")
            self.abort_request(request_id)
    
    def process_kv_response(self,
                            global_ranks: List[int],
                            kv_response: KvPreparedResponse) -> None:
        """Process a request output from the engine"""
        request_id = kv_response.request_id
        kv_response.global_ranks = global_ranks
        self._request_streams.get(request_id).put(kv_response)
        if kv_response.error !=0:
            self.abort_request(request_id)
    
    def process_exception(self,
                          request_id: str,
                          exception: Exception,
                          *,
                          verbose: bool = False) -> None:
        """Propagate an exception from the engine."""
        self._request_streams[request_id].put(exception)
        if verbose:
            logger.info(f"Finished request {request_id}.")
        self.abort_request(request_id)
        
    def add_request(self, request_id: str,
                    **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")
        stream = AsyncStream(request_id)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
            **engine_add_request_kwargs
        }))
        self.new_requests_event.set()
        return stream

    def add_kv_response(self,
                        **engine_kv_response_kwargs) -> None:
        self._kv_responses.put_nowait({
            **engine_kv_response_kwargs
        })
        self.new_requests_event.set()
        
    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info(f"Aborted request {request_id}.")

        self._finished_requests.put_nowait(request_id)

        if request_id not in self._request_streams or self._request_streams[
                request_id].finished:
            # The request has already finished or been aborted.
            return

        self._request_streams[request_id].finish()
        
    def get_kv_responses(self) -> List[dict]:
        kv_responses: List = []
        while not self._kv_responses.empty():
            response = self._kv_responses.get_nowait()
            kv_responses.append(response)
        return kv_responses    
    
    def get_new_and_finished_requests(self) -> Tuple[List[Dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)
            
        return new_requests, finished_requests

    async def wait_for_new_requests(self):
        if not self.has_new_requests():
            await self.new_requests_event.wait()
        self.new_requests_event.clear()

    def has_new_requests(self):
        return not self._new_requests.empty()


class AsyncSched:
    def __init__(self, 
                 instances_num: int,
                 start_engine_loop: bool = True,
                 static_dop: Optional[int] = 8,
                 window_size: Optional[int] = 10):
        self.video_sched = VideoSched(instances_num = instances_num,
                                       static_dop = static_dop,
                                       window_size = window_size)
        self.start_engine_loop = start_engine_loop
        
        self.background_loop = None
        self._errored_with: Optional[BaseException] = None
        
        self.task_queue = queue.Queue()
        self.consumers = []
    
    def post_http_request(self, pload, api_url) -> requests.Response:
        headers = {"User-Agent": "Test Client"}

        response = requests.post(api_url, headers=headers, json=pload)
        return response

    def update_requests_cur_steps(self, request_id, cur_step):
        self.video_sched.update_requests_cur_steps(request_id, cur_step)
    
    def process2(self) -> None:
        while True:
            #if self.task_queue.empty():
            #    continue
            task = self.task_queue.get()
            print(f"request {task.request_id} resolution {task.resolution} worker ids {task.worker_ids}")
            #api_url = "http://127.0.0.1:8000/async_generate"
            api_url = "http://127.0.0.1:8000/async_generate_dit"
            pload = {
                "request_id": task.request_id,
                "prompt": task.prompt,
                "resolution": task.resolution, 
                "aspect_ratio": task.aspect_ratio,
                "num_frames": task.num_frames,
                "worker_ids": task.worker_ids,
            }
            _ = self.post_http_request(pload = pload, api_url = api_url)
            #self.video_sched.scheduler.breakdown_update_gpu_status(group_id = task.request_id, last = False)

            api_url2 = "http://127.0.0.1:8000/async_generate_vae"
            pload = {
                "request_id": task.request_id,
                "worker_ids": task.worker_ids, #[task.worker_ids[0]]
            }
            _ = self.post_http_request(pload=pload, api_url=api_url2)

            #self.video_sched.scheduler.breakdown_update_gpu_status(group_id = task.request_id, last = True)
            self.video_sched.scheduler.naive_update_gpu_status(group_id=task.request_id)
            #self.video_sched.scheduler.window_update_gpu_status(group_id=task.request_id)
            #self.video_sched.scheduler.naive_baseline_update_gpu_status(resolution = task.resolution, worker_ids = task.worker_ids)
            #self.video_sched.scheduler.smart_baseline_update_gpu_status(worker_ids = task.worker_ids, res = task.resolution) #req_id = task.request_id)
    
    def process(self,):
        while True:
            #if self.task_queue.empty():
            #    continue
            task = self.task_queue.get()  # 阻塞，直到有任务
            #if task is None:
            #    break  # 如果任务是 None，表示结束
            #with open("/data/home/scyb091/VideoSys/examples/global_scheduler/gen.txt", "a") as file:
            #        file.write(f"{task.request_id} {time.time()}\n")
            print(f"request {task.request_id} resolution {task.resolution} dit's worker ids {task.worker_ids}")

            '''if task.worker_ids == "144p":
                api_url = "http://127.0.0.1:8000/async_generate"
                pload = {
                    "request_id": task.request_id,
                    "prompt": task.prompt,
                    "resolution": task.resolution, 
                    "aspect_ratio": task.aspect_ratio,
                    "num_frames": task.num_frames,
                    "worker_ids": task.worker_ids,
                }
                _ = self.post_http_request(pload=pload, api_url=api_url)
                #self.video_sched.scheduler.update_gpu_status(last = True, group_id = task.request_id, sjf = True)
                self.video_sched.scheduler.update_gpu_status(last = True, group_id = task.request_id, sjf = False)
            else:'''
            api_url = "http://127.0.0.1:8000/async_generate_dit"
            pload = {
                "request_id": task.request_id,
                "prompt": task.prompt,
                "resolution": task.resolution, 
                "aspect_ratio": task.aspect_ratio,
                "num_frames": task.num_frames,
                "worker_ids": task.worker_ids,
            }
            _ = self.post_http_request(pload=pload, api_url=api_url)
                    #self.video_sched.scheduler.update_gpu_status(last = False, group_id = task.request_id, sjf = True)
            self.video_sched.scheduler.update_gpu_status(last = False, group_id = task.request_id, sjf = False)
            #self.video_sched.scheduler.update_gpu_status(last = False, group_id = task.request_id, sjf = True)
                #self.video_sched.scheduler.breakdown_update_gpu_status(group_id = task.request_id, last = False)
            print(f"request {task.request_id} resolution {task.resolution} vae's worker ids {task.worker_ids[0]}") #task.worker_ids
            api_url = "http://127.0.0.1:8000/async_generate_vae"
            pload = {
                "request_id": task.request_id,
                "worker_ids": [task.worker_ids[0]], #task.worker_ids
            }
            _ = self.post_http_request(pload=pload, api_url=api_url)
                    #self.video_sched.scheduler.update_gpu_status(last = True, group_id = task.request_id, sjf = True)
            self.video_sched.scheduler.update_gpu_status(last = True, group_id = task.request_id, sjf = False)
            #self.video_sched.scheduler.update_gpu_status(last = True, group_id = task.request_id, sjf = True)
                #self.video_sched.scheduler.breakdown_update_gpu_status(group_id = task.request_id, last = True)
        return 
    
    def create_consumer(self, instances_num: int):
        for _ in range(instances_num):
            #consumer = threading.Thread(target=self.process)
            consumer = threading.Thread(target=self.process2)
            consumer.daemon = True
            consumer.start()
            self.consumers.append(consumer)
        
    def destory_consumer(self):
        for consumer in self.consumers:
            consumer.join()

    async def run_engine_loop(self):
        has_requests_in_progress = False
        while True:
            if (not has_requests_in_progress 
                and not self.video_sched.scheduler.waiting 
                ):
                
                await self._request_tracker.wait_for_new_requests()
            # Abort if iteration takes too long due to unrecoverable errors
            # (eg. NCCL timeouts).
            try:
                has_requests_in_progress = await asyncio.wait_for(
                    self.engine_step(), ENGINE_ITERATION_TIMEOUT_S)
                #print("AsyncSched run_engine_loop ", len(self.video_sched.scheduler.waiting))
            except asyncio.TimeoutError as exc:
                raise
            await asyncio.sleep(0)

    # async def async_send_to(self, entry_point: Tuple[str, int], func_name: str, data: CommData):
    #     api_url = f"http://{entry_point[0]}:{entry_point[1]}/{func_name}"
    #     async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
    #         async with session.post(url=api_url, json=data.payload,
    #                                 headers=data.headers) as response:
    #             return await response.json()
            
    # async def send_to_worker(self, seq_group: SequenceGroup):
    #     entry_point = ("127.0.0.1", 8000)
    #     data = CommData(seq_group.__json__())
    #     await self.async_send_to(entry_point, "async_generate", seq_group)
    #     return
        
    async def step_async(self):
        #t1 = time.time()
        #seq_group = self.video_sched.scheduler.hungry_first_priority_schedule()
        #t2 = time.time()
        #with open("costs.txt", "a") as file:    
        #    file.write(f"{t2-t1}\n") 
        #seq_group = self.video_sched.scheduler.least_remaining_time_schedule()
        #seq_group = self.video_sched.scheduler.naive_baseline_schedule()
        #seq_group = self.video_sched.scheduler.naive_baseline_greedy_schedule()
        #seq_group = self.video_sched.scheduler.naive_partition_schedule()
        #seq_group = self.video_sched.scheduler.smart_static_partition_schedule()
        #seq_group = self.video_sched.scheduler.smart_dynamic_partition_schedule()
        #seq_group = self.video_sched.scheduler.sjf_priority_schedule()
        #seq_group = self.video_sched.scheduler.continuous_batching_schedule()
        seq_group = self.video_sched.scheduler.window_based_sjf_schedule()
        #seq_group = self.video_sched.scheduler.window_based_sjf_with_hungry_update_schedule()
        #seq_group = self.video_sched.scheduler.fcfs_decouple_sjf_schedule()
        #seq_groups = self.video_sched.scheduler.window_based_sjf_with_sjf_update_schedule()
        #if seq_groups:
        #    for seq_group in seq_groups:
        if seq_group:
            self.task_queue.put(seq_group)
            #return True
        return None
     
    async def engine_step(self) -> bool:
        new_requests, _ = (
            self._request_tracker.get_new_and_finished_requests())        
        for new_request in new_requests:
            self.video_sched.add_request(**new_request)
        request_outputs = await self.step_async()
        return request_outputs != None
    
    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
            and not self._background_loop_unshielded.done())
        
    @property
    def is_stopped(self) -> bool:
        return self.errored or (self.background_loop is not None
                                and self._background_loop_unshielded.done())

    @property
    def errored(self) -> bool:
        return self._errored_with is not None
 
    def set_errored(self, exc: Exception) -> None:
        self._errored_with = exc

    def _error_callback(self, exc: Exception) -> None:
        self.set_errored(exc)
        self._request_tracker.propagate_exception(exc)
        
    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.errored:
            raise AsyncEngineDeadError(
                "Background loop has errored already.") from self._errored_with
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        # Initialize the RequestTracker here so it uses the right event loop.
        self._request_tracker = RequestTracker()

        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.run_engine_loop())
        self._background_loop_unshielded.add_done_callback(
            partial(_raise_exception_on_finish,
                    error_callback=self._error_callback))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)
        
    async def generate(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        num_frames: Optional[str] = None,
    ) -> AsyncStream:
        if not self.is_running:
            if self.start_engine_loop:
                self.start_engine_time = time.time()
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")

        stream = await self.add_request(
            request_id = request_id,
            prompt = prompt,
            resolution = resolution,
            aspect_ratio = aspect_ratio,
            num_frames = num_frames,
        )

        # async for request_output in stream:
        #     yield request_output

    async def add_request(self, 
        request_id: str,
        prompt: Optional[str] = None,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        num_frames: Optional[str] = None):
        
        stream = self._request_tracker.add_request(
            request_id = request_id,
            prompt = prompt,
            resolution = resolution,
            aspect_ratio = aspect_ratio,
            num_frames = num_frames,
        )
        return stream

class AsyncEngine:
    def __init__(self, 
                 config: OpenSoraConfig,
                 deploy_config: DeployConfig,
                 start_engine_loop: bool = True,):
        self.config = config
        self.deploy_config = deploy_config
        self.parallel_worker_tasks = None
        self.video_engine = VideoSysEngine(config=self.config, deploy_config=self.deploy_config)
        self.start_engine_loop = start_engine_loop
        
        self.background_loop = None
        self._errored_with: Optional[BaseException] = None
        self.request_workers = {}

        self.gs_url = "http://127.0.0.1:8001/update_cur_step"
        self.update_cur_step_tasks: Queue[Tuple[str, int]] = Queue()
    
    def post_http_request(self, pload, api_url) -> requests.Response:
        headers = {"User-Agent": "Test Client"}
        response = requests.post(api_url, headers=headers, json=pload)
        return response
    
    async def run_engine_loop(self):
        has_requests_in_progress = False
        print("run_engine_loop ")
        while True:
            if (not has_requests_in_progress 
                and not self.video_engine.scheduler.waiting 
                and not self.video_engine.scheduler.send_transfering
                and not self.video_engine.scheduler.recv_transfering
                ):
                
                await self._request_tracker.wait_for_new_requests()
            # Abort if iteration takes too long due to unrecoverable errors
            # (eg. NCCL timeouts).
            try:
                has_requests_in_progress = await asyncio.wait_for(
                    self.engine_step(), ENGINE_ITERATION_TIMEOUT_S)
            except asyncio.TimeoutError as exc:
                raise
            await asyncio.sleep(0)
    
    async def step_async(self):
        if self.config.enable_separate:
            send_finished_reqs = self.video_engine.scheduler._check_tranfer_finished_req()
            if send_finished_reqs:
                self.video_engine.remove_dit(send_finished_reqs=send_finished_reqs)
        seq_group = self.video_engine.scheduler.schedule()
        if seq_group:
            if not self.config.enable_separate:
                video = self.video_engine.generate(prompt=seq_group.prompt,
                    resolution=seq_group.resolution,
                    aspect_ratio=seq_group.aspect_ratio,
                    num_frames=seq_group.num_frames,
                ).video[0]
                print("video info ", type(video), video.shape)
                shape = None
                self.video_engine.save_video(video, f"./outputs/{seq_group.prompt}.mp4")
            else:
                if self.config.worker_type == "dit":
                    t1 = time.time()
                    request_id, shape = self.video_engine.generate_dit(request_id=seq_group.request_id, 
                                                        prompt=seq_group.prompt,
                                                        resolution=seq_group.resolution,
                                                        aspect_ratio=seq_group.aspect_ratio,
                                                        num_frames=seq_group.num_frames,
                                                        )
                    print("dit request_id, shape ", request_id, shape)
                    seq_group.shape = shape
                    self.video_engine.scheduler.add_send_transfering(seq_group)
                    
                    t2 = time.time()
                    self.video_engine.transfer_dit(request_id=seq_group.request_id)
                    t3 = time.time()
                    print("step async ", t3-t2, t2-t1)
                else:
                    print("vae request_id ", time.time(), seq_group.request_id)
                    t1 = time.time()
                    video = self.video_engine.generate_vae(request_id=seq_group.request_id).video[0]
                    t2 = time.time()
                    print("video step async ", time.time(), t2-t1, type(video), video.shape)
                    self.video_engine.save_video(video, f"./outputs/{seq_group.prompt}.mp4")
            return RequestOutput(seq_group.request_id, seq_group.prompt, seq_group.shape, True)
        return None

    async def trans_kv_step_aysnc(self):
        if not self.config.enable_separate:
            return
        if not self.video_engine.scheduler.send_transfering and not self.video_engine.scheduler.recv_transfering:
            return 

        finished_work_tasks = self.video_engine.get_finished_transfer_tasks()
        for finished_tasks in finished_work_tasks:
            print("finished_work_tasks ", finished_work_tasks)
            for worker_finished_task in finished_tasks:
                send_finished_tasks = []
                recv_finished_tasks = []
                # for finished_task in worker_finished_task[0]:
                    # send_finished_tasks.append(trans_ops.TransferTaskMeta.deserialize(finished_task))
                # for finished_task in worker_finished_task[1]:
                    # recv_finished_tasks.append(trans_ops.TransferTaskMeta.deserialize(finished_task))
                # real_send_finished_req_ids = self.video_engine.send_kv_trans_scheduler.add_finished_tasks(send_finished_tasks)
                # real_recv_finished_req_ids = self.video_engine.recv_kv_trans_scheduler.add_finished_tasks(recv_finished_tasks)
                # if real_send_finished_req_ids:
                #     self.video_engine.scheduler.add_send_finished(real_send_finished_req_ids)
                # if real_recv_finished_req_ids:
                #     self.video_engine.scheduler.add_recv_finished(real_recv_finished_req_ids)

        send_tasks = self.video_engine.send_kv_trans_scheduler.schedule()
        recv_tasks = self.video_engine.recv_kv_trans_scheduler.schedule()
        
        if send_tasks or recv_tasks:
            self.video_engine.trans_blocks(
                send_tasks = send_tasks,
                recv_tasks = recv_tasks,
            )
            
    async def engine_step(self) -> bool:
        new_requests, finished_requests = (
            self._request_tracker.get_new_and_finished_requests())
        # print("engine_step ")
        
        for new_request in new_requests:
            print("new_request ", new_request)
            self.video_engine.add_request(**new_request)
        
        if self.config.enable_separate:
            kv_responses = self._request_tracker.get_kv_responses()
            for kv_response in kv_responses:
                # Add the response
                self.video_engine.add_kv_response(**kv_response)
                    
            #kv_responses out, receiver process allocate kv cache req from sender, and return allocat kv num
            kv_responses = self.video_engine.schedule_vae_waiting()
            for kv_response in kv_responses:
                self._request_tracker.process_kv_response(
                    self.video_engine.get_global_ranks(), kv_response)
                
            await self.trans_kv_step_aysnc()

        request_outputs = await self.step_async()

        if request_outputs:
            self._request_tracker.process_request_output(self.video_engine.get_global_ranks(), request_outputs)


        return request_outputs!=None
    
    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
            and not self._background_loop_unshielded.done())
        
    @property
    def is_stopped(self) -> bool:
        return self.errored or (self.background_loop is not None
                                and self._background_loop_unshielded.done())

    @property
    def errored(self) -> bool:
        return self._errored_with is not None
 
    def set_errored(self, exc: Exception) -> None:
        self._errored_with = exc

    def _error_callback(self, exc: Exception) -> None:
        self.set_errored(exc)
        self._request_tracker.propagate_exception(exc)
        
    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.errored:
            raise AsyncEngineDeadError(
                "Background loop has errored already.") from self._errored_with
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        # Initialize the RequestTracker here so it uses the right event loop.
        self._request_tracker = RequestTracker()

        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.run_engine_loop())
        self._background_loop_unshielded.add_done_callback(
            partial(_raise_exception_on_finish,
                    error_callback=self._error_callback))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)
        
    async def generate(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        num_frames: Optional[str] = None,
        shape: Optional[List] = None,
        global_ranks: Optional[List] = None
    ) -> AsyncStream:
        if not self.is_running:
            if self.start_engine_loop:
                self.start_engine_time = time.time()
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")

        stream = await self.add_request(
            request_id = request_id,
            prompt = prompt,
            resolution = resolution,
            aspect_ratio = aspect_ratio,
            num_frames = num_frames,
            shape = shape,
            global_ranks = global_ranks,
        )

        async for request_output in stream:
            yield request_output

    async def add_request(self, 
        request_id: str,
        prompt: Optional[str] = None,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        num_frames: Optional[str] = None,
        shape: Optional[List] = None,
        global_ranks: Optional[List] = None):
        
        stream = self._request_tracker.add_request(
            request_id = request_id,
            prompt = prompt,
            resolution = resolution,
            aspect_ratio = aspect_ratio,
            num_frames = num_frames,
            shape = shape,
            global_ranks = global_ranks
        )
        return stream
    

    async def get_nccl_id(self, dst_channel, worker_type):
        nccl_id = self.video_engine.get_nccl_id(dst_channel, worker_type)
        res = self.video_engine.create_comm(nccl_id=nccl_id, dst_channel=dst_channel, worker_type=worker_type)
        return nccl_id
        
    
    async def create_comm(self, nccl_id, dst_channel, worker_type) -> None:
        if worker_type == "vae":
            res = self.video_engine.create_comm(nccl_id=nccl_id, dst_channel=dst_channel, worker_type="vae")
        else:
            res = self.video_engine.create_comm(nccl_id=nccl_id, dst_channel=dst_channel, worker_type="dit")
    
    async def add_kv_response(
        self,
        response: KvPreparedResponse,
    ) -> None:
        self._request_tracker.add_kv_response(response=response)
    
    async def build_conn(self, rank, world_size, group_name) -> None:
        self.video_engine._build_conn(rank, world_size, group_name)
    
    async def build_worker_comm(self, worker_ids):
        await self.video_engine.build_worker_comm(worker_ids)

    async def build_worker_comm_data(self, worker_ids):
        await self.video_engine.build_worker_comm_data(worker_ids)
        
    
    async def worker_generate_homo(self, worker_ids, request_id, prompt, resolution, aspect_ratio, num_frames) -> None:
        outputs = await self.video_engine.async_generate(worker_ids=worker_ids, prompt=prompt, resolution=resolution, aspect_ratio=aspect_ratio, num_frames=num_frames)
        return outputs.video

    async def worker_generate(self, worker_ids, request_id, prompt, resolution, aspect_ratio, num_frames) -> None:
        # await self.video_engine.async_generate(worker_ids=worker_ids, prompt=prompt,
        #             resolution=resolution,
        #             aspect_ratio=aspect_ratio,
        #             num_frames=num_frames,)
        
        await self.video_engine.prepare_generate(worker_ids=worker_ids,
            prompt=prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            num_frames=num_frames,
        )
        t1 = time.time()
        for index in range(self.video_engine.config.num_sampling_steps):
            #use request_id check sched req to and get worker_ids, if true, need rebuild comm and trans data.
            # new worker need execute prepare_generate
            #
            if request_id in self.request_workers:
            #     print("no new gpus ", request_id)
            # else:
                # print("new gpus ", request_id, self.request_workers[request_id])
                # await self.destory_worker_comm(worker_ids=worker_ids)
                new_worker_ids = self.request_workers[request_id]
                pre_worker_ids = list(set(new_worker_ids) - set(worker_ids))
                await self.video_engine.prepare_generate(
                    worker_ids=pre_worker_ids,
                    prompt=prompt,
                    resolution=resolution,
                    aspect_ratio=aspect_ratio,
                    num_frames=num_frames,
                )
                print("new_worker_ids, worker_ids ", new_worker_ids, worker_ids)
                worker_ids = copy.deepcopy(new_worker_ids)
                await self.build_worker_comm(worker_ids)
                del self.request_workers[request_id]
                
            await self.video_engine.index_iteration_generate(worker_ids=worker_ids, i=index)
            pload = {
                "request_id": request_id,
                "cur_step": index + 1,
            }
            api_url = "http://127.0.0.1:8001/update_cur_step"
            self.post_http_request(pload=pload, api_url=api_url)
        t2 = time.time()
        print("t2-t1 " , t2-t1)

    def update_requests_cur_steps(self) -> None:
        while True:
            #if self.update_cur_step_tasks.empty():
            #    continue
            request_id, cur_step = self.update_cur_step_tasks.get()
            pload = {
                "request_id": request_id,
                "cur_step": cur_step,
            }
            _ = self.post_http_request(pload = pload, api_url = self.gs_url)
            
    def create_update_threads(self, instances_num: int) -> None:
        for _ in range(instances_num): # can't be sure if we need instances_num's threads, may be one is enough
            cur_thread = threading.Thread(target = self.update_requests_cur_steps)
            cur_thread.daemon = True
            cur_thread.start()
    
    async def worker_generate_dit(self, worker_ids, request_id, prompt, resolution, aspect_ratio, num_frames) -> None:
        await self.video_engine.prepare_generate(
            worker_ids=worker_ids,
            prompt=prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            num_frames=num_frames,
        )            
        #t1 = time.time()
        for index in range(self.video_engine.config.num_sampling_steps):
            #use request_id check sched req to and get worker_ids, if true, need rebuild comm and trans data.
            # new worker need execute prepare_generate
            #
            # if request_id not in self.request_workers:
            #     print("no new gpus ", request_id)
            # else:
            #     print("new gpus ", request_id, self.request_workers[request_id])
            if request_id in self.request_workers:
                #if len(self.request_workers[request_id]) % 2 != 0:
                #    self.request_workers[request_id].pop()
                pre_worker_ids = list(set(self.request_workers[request_id]) - set(worker_ids))
            else:
                pre_worker_ids = []
            if pre_worker_ids:
                # await self.destory_worker_comm(worker_ids=worker_ids)
                #if len(pre_worker_ids) % 2 != 0:
                #    pre_worker_ids.pop()
                await self.video_engine.prepare_generate(
                        worker_ids=pre_worker_ids,
                        prompt=prompt,
                        resolution=resolution,
                        aspect_ratio=aspect_ratio,
                        num_frames=num_frames,
                    )
                print(f"request {request_id} resolution {resolution} new worker ids {self.request_workers[request_id]} old worker ids {worker_ids}")
                worker_ids = copy.deepcopy(self.request_workers[request_id])
                if len(worker_ids) % 2 != 0:
                    worker_ids.pop()
                await self.build_worker_comm(worker_ids)
            '''if request_id in self.request_workers:
                #     print("no new gpus ", request_id)
                # else:
                    # print("new gpus ", request_id, self.request_workers[request_id])
                    await self.destory_worker_comm(worker_ids=worker_ids)
                    new_worker_ids = self.request_workers[request_id]
                    pre_worker_ids = list(set(new_worker_ids) - set(worker_ids))
                    await self.video_engine.prepare_generate(
                        worker_ids=pre_worker_ids,
                        prompt=prompt,
                        resolution=resolution,
                        aspect_ratio=aspect_ratio,
                        num_frames=num_frames,
                    )
                    print("new_worker_ids, worker_ids ", new_worker_ids, worker_ids)
                    worker_ids = copy.deepcopy(new_worker_ids)
                    await self.build_worker_comm(worker_ids)
                    del self.request_workers[request_id]'''
            self.update_cur_step_tasks.put((request_id, index + 1)) # comnunicate first then the scheduler will re-allocate while worker exectuing
            await self.video_engine.index_iteration_generate(worker_ids = worker_ids, i = index)
        #t2 = time.time()
        #print("t2-t1 " , t2-t1)
        # await self.video_engine.async_generate_dit(worker_ids=worker_ids, request_id=request_id, prompt=prompt,
        #             resolution=resolution,
        #             aspect_ratio=aspect_ratio,
        #             num_frames=num_frames,)

    async def worker_generate_vae(self, worker_ids, request_id) -> None:
        await self.video_engine.async_generate_vae(worker_ids=worker_ids, request_id=request_id)
    
    async def worker_generate_vae_step(self, worker_ids, request_id) -> None:
        # await self.video_engine.worker_generate_vae_step(worker_ids=worker_ids)
        output = await self.video_engine.worker_generate_vae_step(worker_ids=worker_ids)
        return output.video
        
    async def update_request_workers(self, request_id, worker_ids) -> None:
        self.request_workers[request_id] = worker_ids
        
    async def destory_worker_comm(self, worker_ids):
        await self.video_engine.destory_worker_comm(worker_ids)

        