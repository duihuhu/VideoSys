from typing import Optional, List, Dict

class RequestOutput:
    """The output data of a request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
        finished: Whether the whole request is finished.
    """

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        shape: Optional[List] = None,
        finished: Optional[bool] = None,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.shape = shape
        self.finished = finished
        self.global_ranks: List[int] = None
        
    def __json__(self) -> Dict:
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "shape": self.shape,
            "finished": self.finished
        }


class KvPreparedResponse:
    def __init__(
        self,
        request_id: str,
        error: int,
        error_msg: str,
        computed_blocks: int,
        transfer_tag: str,
        dst_cpu_blocks: Optional[List[int]] = None,
        has_dram: Optional[bool] = False
    ) -> None:
        self.request_id = request_id
        self.error = error
        self.error_msg = error_msg
        self.global_ranks = None
        self.transfer_tag = transfer_tag
    def __json__(self) -> Dict:
        return {
            "request_id": self.request_id,
            "global_ranks": self.global_ranks,
            "error": self.error,
            "error_msg": self.error_msg,
            "transfer_tag": self.transfer_tag,
        }