from typing import Optional, List
class SequenceGroup:
    """A group of sequences that are generated from the same prompt.

    Args:
        request_id: The ID of the request.
    """

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        num_frames: Optional[str] = None,
        shape: Optional[List] = None,
        worker_ids: Optional[List] = None,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio
        self.num_frames = num_frames
        self.shape = shape
        self.worker_ids = worker_ids
    
    def __json__(self):
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "resolution": self.resolution,
            "aspect_ratio": self.aspect_ratio,
            "num_frames": self.num_frames,
            "worker_ids": self.worker_ids
        }
        
