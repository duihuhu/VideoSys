from typing import Optional

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
        finished: Optional[bool],
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.finished = finished

    def __repr__(self) -> str:
        return (f"RequestOutput(request_id={self.request_id}, "
                f"prompt={self.prompt!r}, "
                f"finished={self.finished}")