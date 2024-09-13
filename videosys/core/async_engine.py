from videosys import VideoSysEngine
import asyncio
import os
from typing import (Callable, Dict, List, Optional, Set, Tuple, Union)
from functools import partial
import time
from videosys.utils.logging import logger
from videosys.core.outputs import RequestOutput

ENGINE_ITERATION_TIMEOUT_S = int(
    os.environ.get("VLLM_ENGINE_ITERATION_TIMEOUT_S", "60"))

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
                               request_output: RequestOutput,
                               *,
                               verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id
        self._request_streams[request_id].put(request_output)
        if request_output.finished:
            logger.info(f"Finished request {request_id}.")
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

class AsyncEngine:
    def __init__(self, config,
                 start_engine_loop: bool = True,):
        self.config = config
        self.parallel_worker_tasks = None
        self.video_engine = VideoSysEngine(config=self.config)
        self.start_engine_loop = start_engine_loop
        
        self.background_loop = None
        self._errored_with: Optional[BaseException] = None

    async def run_engine_loop(self):
        has_requests_in_progress = False
        print("run_engine_loop ")
        while True:
            if (not has_requests_in_progress 
                and not self.video_engine.scheduler.waiting 
                # and not self.engine.scheduler.send_transfering
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
        seq_group = self.video_engine.scheduler.schedule()
        if seq_group:
            video = self.video_engine.generate(prompt=seq_group.prompt,
                resolution=seq_group.resolution,
                aspect_ratio=seq_group.aspect_ratio,
                num_frames=seq_group.num_frames,
            ).video[0]
            print(type(video))
            self.video_engine.save_video(video, f"./outputs/{seq_group.prompt}.mp4")
            return RequestOutput(seq_group.request_id, seq_group.prompt, True)
        return None


    async def engine_step(self) -> bool:
        new_requests, finished_requests = (
            self._request_tracker.get_new_and_finished_requests())
        print("engine_step ")
        
        for new_request in new_requests:
            print("new_request ", new_request)
            self.video_engine.add_request(**new_request)
            
        request_outputs = await self.step_async()
        if request_outputs:
            self._request_tracker.process_request_output(request_outputs)
        
        
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
        prompt: Optional[str],
        resolution: Optional[str],
        aspect_ratio: Optional[str],
        num_frames: Optional[str],
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
            num_frames = num_frames
        )

        async for request_output in stream:
            yield request_output

    async def add_request(self, 
        request_id: str,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        num_frames: str):
        
        stream = self._request_tracker.add_request(
            request_id = request_id,
            prompt = prompt,
            resolution = resolution,
            aspect_ratio = aspect_ratio,
            num_frames = num_frames
        )
        return stream
    
