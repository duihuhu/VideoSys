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
        prompt: Optional[str],
        shape: Optional[List],
        finished: Optional[bool],
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.shape = shape
        self.finished = finished
        
    def __json__(self) -> Dict:
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "shape": self.shape,
            "finished": self.finished
        }