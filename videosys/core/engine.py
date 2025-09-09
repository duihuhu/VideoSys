import os
from functools import partial

import torch

import videosys

from .mp_utils import ProcessWorkerWrapper, ResultHandler, WorkerMonitor, get_distributed_init_method, get_open_port
from videosys.core.sequence import SequenceGroup
from videosys.core.scheduler import Scheduler, VideoScheduler
from videosys.core.kv_trans_scheduler import SendKvTransferScheduler, RecvKvTransScheduler
from videosys.core.outputs import KvPreparedResponse
from typing import (Any, Awaitable, Callable, TypeVar, Optional, List)
import asyncio
T = TypeVar("T")

class VideoSched:
    def __init__(self, instances_num: int):
        self.scheduler = VideoScheduler(instances_num = instances_num)
        
    def add_request(self, 
                request_id,
                prompt: Optional[str] = None,
                resolution: Optional[str] = None,
                aspect_ratio: Optional[str] = None,
                num_frames: Optional[str] = None):
        seq_group = SequenceGroup(request_id=request_id, prompt=prompt, resolution=resolution,\
            aspect_ratio=aspect_ratio, num_frames=num_frames)
        #print(f"add request {request_id} resolution {resolution}")
        self.scheduler.add_seq_group(seq_group)
    
    def update_requests_cur_steps(self, request_id, cur_step):
        self.scheduler.update_requests_cur_steps(request_id, cur_step)
        
class VideoSysEngine:
    """
    this is partly inspired by vllm
    """

    def __init__(self, config, deploy_config = None):
        self.config = config
        self.deploy_config = deploy_config
        self.parallel_worker_tasks = None
        self.scheduler = Scheduler()
        # if config.worker_type=="dit":
        self.send_kv_trans_scheduler = SendKvTransferScheduler(1, config.worker_type)
        # else:
        self.recv_kv_trans_scheduler = RecvKvTransScheduler(1, config.worker_type)

        self._init_worker(config.pipeline_cls)
        self._init_all_process_group()

    def _init_worker(self, pipeline_cls):
        world_size = self.config.num_gpus


        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(self.config.dworld_size))
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(world_size))

        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        # Set OMP_NUM_THREADS to 1 if it is not set explicitly, avoids CPU
        # contention amongst the shards
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "1"

        # NOTE: The two following lines need adaption for multi-node
        assert world_size <= torch.cuda.device_count()

        # change addr for multi-node

        if world_size == 1:
            self.workers = []
            self.worker_monitor = None      
              
            driver_result_handler = ResultHandler()
            distributed_init_method = get_distributed_init_method("127.0.0.1", get_open_port())
            self.driver_worker = ProcessWorkerWrapper(
                        driver_result_handler,
                        partial(
                            self._create_pipeline,
                            pipeline_cls=pipeline_cls,
                            rank=0,
                            local_rank=0,
                            world_size=world_size,
                            distributed_init_method=distributed_init_method,
                        ),
                    )
            self.dirver_worker_monitor = WorkerMonitor([self.driver_worker], driver_result_handler)
            self.workers.append(self.driver_worker)
            driver_result_handler.start()
            self.dirver_worker_monitor.start()
        else:
            distributed_init_method = get_distributed_init_method("127.0.0.1", get_open_port())
            result_handler = ResultHandler()
            self.workers = [
                ProcessWorkerWrapper(
                    result_handler,
                    rank,
                    partial(
                        self._create_pipeline,
                        pipeline_cls=pipeline_cls,
                        rank=rank,
                        local_rank=rank,
                        world_size=world_size,
                        distributed_init_method=distributed_init_method,
                    ),
                )
                for rank in range(0, world_size)
            ]

            self.worker_monitor = WorkerMonitor(self.workers, result_handler)
            result_handler.start()
            self.worker_monitor.start()

        # self.driver_worker = self._create_pipeline(
        #     pipeline_cls=pipeline_cls, distributed_init_method=distributed_init_method
        # )
        


    def get_physical_device_id(self, rank):
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None:
            # 如果 CUDA_VISIBLE_DEVICES 未设置，逻辑设备即为物理设备
            return rank
        # CUDA_VISIBLE_DEVICES 是一个以逗号分隔的字符串
        print("cuda_visible_devices ", cuda_visible_devices)
        devices = cuda_visible_devices.split(',')
        return int(devices[rank])

    # TODO: add more options here for pipeline, or wrap all options into config
    def _create_pipeline(self, pipeline_cls, rank=0, local_rank=0, world_size=0, distributed_init_method=None):
        # videosys.initialize(rank=0, local_rank=local_rank, world_size=1, init_method=distributed_init_method, seed=42)
        videosys.initialize_device(local_rank=local_rank, world_size= world_size, distributed_init_method = distributed_init_method)
        pipeline = pipeline_cls(self.config)
        return pipeline

    def _build_conn(self, rank, world_size, group_name, distributed_init_method="tcp://127.0.0.1:41377"):
        self.config.num_gpus = world_size
        videosys.initialize(rank=rank, world_size=self.config.num_gpus, init_method=distributed_init_method, seed=42)
        
        
    def _run_workers(
        self,
        method: str,
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""

        # Start the workers first.
        worker_outputs = [worker.execute_method(method, *args, **kwargs) for worker in self.workers]

        if async_run_tensor_parallel_workers_only:
            # Just return futures
            return worker_outputs

        driver_worker_method = getattr(self.driver_worker, method)
        driver_worker_output = driver_worker_method(*args, **kwargs)

        # Get the results of the workers.
        return [driver_worker_output] + [output.get() for output in worker_outputs]

    def _run_workers_without_driver_worker(
        self,
        method: str,
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""

        # Start the workers first.
        worker_outputs = [worker.execute_method(method, *args, **kwargs) for worker in self.workers]

        if async_run_tensor_parallel_workers_only:
            # Just return futures
            return worker_outputs
        # Get the results of the workers.
        return [output.get() for output in worker_outputs]
    

    def _init_all_process_group(self):
        res = self._run_workers_without_driver_worker("init_all_process_group")
        return res

    async def _set_curr_parallel_mgr(
        self,
        worker_ids: List[int],
    ) -> Any:
        res = await self._run_workers_by_id_async(worker_ids, "set_curr_parallel_mgr", worker_ids)
        return res
        
    async def _run_workers_async(
        self,
        method: str, 
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,):
        
        worker_outputs = [worker.execute_method_async(method, *args, **kwargs) for worker in self.workers]
        if async_run_tensor_parallel_workers_only:
            # Just return futures
            return worker_outputs
        results = await asyncio.gather(*worker_outputs)
        return [results]

    async def _run_workers_by_id_async(
        self,
        worker_ids: List[int],
        method: str,
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""

        # Start the workers first.
        worker_outputs = [self.workers[id].execute_method_async(method, *args, **kwargs) for id in worker_ids]

        if async_run_tensor_parallel_workers_only:
            # Just return futures
            return worker_outputs
        results = await asyncio.gather(*worker_outputs)
        return [results]
    
    async def _run_workers_dit_aync(self,
        method: str,
        worker_ids,
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,):
        print("run_worker_dit aync")
        # worker_outputs = [worker.execute_method_async(method, *args, **kwargs) for worker in self.workers]
        worker_outputs = []
        for (worker_id, idx) in zip(worker_ids, range(len(worker_ids))):
            if idx == 0:
                kwargs["store_dit"] = True
            else:
                kwargs["store_dit"] = False
            worker_outputs.append(self.workers[worker_id].execute_method_async(method, *args, **kwargs))
                
        if async_run_tensor_parallel_workers_only:
            # Just return futures
            return worker_outputs
        results = await asyncio.gather(*worker_outputs)
        return [results]

    async def _run_workers_vae_aync(self,
        method: str,
        worker_ids,
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,):
        # print("run_worker_vae aync")
        worker_outputs = [self.workers[worker_id].execute_method_async(method, *args, **kwargs) for worker_id in worker_ids]

        if async_run_tensor_parallel_workers_only:
            # Just return futures
            return worker_outputs
        results = await asyncio.gather(*worker_outputs)
        return [results]

    def _driver_execute_model(self, *args, **kwargs):
        return self.driver_worker.generate(*args, **kwargs)

    async def build_worker_comm(self, worker_ids):
        res = await self._set_curr_parallel_mgr(worker_ids)
        return res
        # for worker_id in worker_ids:
        #     self.workers[worker_id].execute_method("build_worker_comm", worker_ids)

    async def build_worker_comm_data(self, worker_ids):
        distributed_init_method = get_distributed_init_method("127.0.0.1", get_open_port())
        for worker_id, rank in zip(worker_ids, range(len(worker_ids))):
            self.workers[worker_id].execute_method("build_worker_comm_data", rank, len(worker_ids), distributed_init_method=distributed_init_method)


    async def destory_worker_comm(self, worker_ids):
        for worker_id in worker_ids:
            self.workers[worker_id].execute_method("destory_worker_comm")
    
    async def async_save_video(self, worker_ids, *args, **kwargs):
        await self._run_workers_by_id_async(worker_ids, "save_video", *args, **kwargs)

    async def async_generate(self, worker_ids, *args, **kwargs):
        outputs = await self._run_workers_by_id_async(worker_ids, "generate", *args, **kwargs)
        return outputs[0]
    
    async def async_generate_dit(self, worker_ids, *args, **kwargs):
        video = await self._run_workers_dit_aync("generate_dit", worker_ids, *args, **kwargs)
        return video[0]

    async def prepare_generate(self, worker_ids, *args, **kwargs):
        await self._run_workers_by_id_async(worker_ids, "prepare_generate", *args, **kwargs)

    async def index_iteration_generate(self, worker_ids, *args, **kwargs):
        await self._run_workers_by_id_async(worker_ids, "index_iteration_generate", *args, **kwargs)
        
    async def async_generate_vae(self, worker_ids, *args, **kwargs):
        video = await self._run_workers_vae_aync("generate_vae", worker_ids, *args, **kwargs)
        return video[0]

    async def worker_generate_vae_step(self, worker_ids, *args, **kwargs):
        video = await self._run_workers_vae_aync("video_generate", worker_ids, *args, **kwargs)
        return video[0]
  
    def generate(self, *args, **kwargs):
        return self._run_workers("generate", *args, **kwargs)[0]

    def generate_dit(self, *args, **kwargs):
        return self._run_workers("generate_dit", *args, **kwargs)[0]
    
    def transfer_dit(self, *args, **kwargs):
        return self.driver_worker.transfer_dit(*args, **kwargs)

    def remove_dit(self, *args, **kwargs):
        return self._run_workers("remove_dit", *args, **kwargs)[0]

    def generate_vae(self, *args, **kwargs):
        return self.driver_worker.generate_vae(*args, **kwargs)

    def get_nccl_id(self, *args, **kwargs):
        return self.driver_worker.get_nccl_id(*args, **kwargs)

    def create_comm(self, *args, **kwargs):
        return self.driver_worker.create_comm(*args, **kwargs)
    
    def allocate_kv(self, *args, **kwargs):
        return self.driver_worker.allocate_kv(*args, **kwargs)
    
    def del_dit_req(self, *args, **kwargs):
        return self._run_workers("del_dit_req", *args, **kwargs)[0]
    
    def fetch_video_addr(self, *args, **kwargs):
        return self.driver_worker.fetch_video_addr(*args, **kwargs)

    def trans_blocks(self, *args, **kwargs):
        return self.driver_worker.trans_blocks(*args, **kwargs)

    def get_finished_transfer_tasks(self, *args, **kwargs):
        return self.driver_worker.get_finished_transfer_tasks()
    
    def stop_remote_worker_execution_loop(self) -> None:
        if self.parallel_worker_tasks is None:
            return

        parallel_worker_tasks = self.parallel_worker_tasks
        self.parallel_worker_tasks = None
        # Ensure that workers exit model loop cleanly
        # (this will raise otherwise)
        self._wait_for_tasks_completion(parallel_worker_tasks)

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        for result in parallel_worker_tasks:
            result.get()

    def save_video(self, video, output_path):
        return self.driver_worker.save_video(video, output_path)

    def shutdown(self):
        if (worker_monitor := getattr(self, "worker_monitor", None)) is not None:
            worker_monitor.close()
        torch.distributed.destroy_process_group()

    def __del__(self):
        self.shutdown()

    def add_request(self, 
                    request_id,
                    prompt: Optional[str] = None,
                    resolution: Optional[str] = None,
                    aspect_ratio: Optional[str] = None,
                    num_frames: Optional[str] = None,
                    shape: Optional[List] = None,
                    global_ranks: Optional[List] = None):
        if self.config.worker_type != "vae":
            seq_group = SequenceGroup(request_id=request_id, prompt=prompt, resolution=resolution,\
                aspect_ratio=aspect_ratio, num_frames=num_frames)
            self.scheduler.add_seq_group(seq_group)
        else:
            seq_group = SequenceGroup(request_id=request_id, prompt=prompt, shape=shape)
            self.scheduler.add_vae_seq_group((seq_group, global_ranks))

    def add_kv_response(self,
        response: KvPreparedResponse
    ) -> None:
        request_id = response.request_id
        if response.error != 0:
            self.scheduler.del_send_transfering(request_id)
            self.del_dit_req(request_id)
            return
        
        video_addr, video_size = self.fetch_video_addr(request_id)
                
        self.send_kv_trans_scheduler.add_kv_request(request_id, response.global_ranks, video_addr, video_size,response.transfer_tag)

                
    def schedule_vae_waiting(self):
        kv_responses = [] 
        while self.scheduler.vae_waiting:
            seq_group = self.scheduler.vae_waiting[0][0]
            global_ranks = self.scheduler.vae_waiting[0][1]
            
            can_allocate, video_addr, video_size = self.allocate_kv(request_id=seq_group.request_id, prompt=seq_group.prompt, shape=seq_group.shape)
            if can_allocate:
                self.scheduler.add_recv_transfering(seq_group)
                transfer_tag = self.recv_kv_trans_scheduler.add_kv_request(seq_group.request_id, global_ranks, video_addr, video_size)
                kv_responses.append(KvPreparedResponse(seq_group.request_id, 0, None, video_addr, transfer_tag))
                
            self.scheduler.vae_waiting.popleft()

        return kv_responses
 
    def get_global_ranks(self):
        return self.config.get_global_ranks()
    
    def make_async(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
        """Take a blocking function, and run it on in an executor thread.

        This function prevents the blocking function from blocking the
        asyncio event loop.
        The code in this function needs to be thread safe.
        """

        def _async_wrapper(*args, **kwargs) -> asyncio.Future:
            loop = asyncio.get_event_loop()
            p_func = partial(func, *args, **kwargs)
            return loop.run_in_executor(executor=None, func=p_func)

        return _async_wrapper
