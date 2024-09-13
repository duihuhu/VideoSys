class SequenceGroup:
    """A group of sequences that are generated from the same prompt.

    Args:
        request_id: The ID of the request.
    """

    def __init__(
        self,
        request_id: str,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        num_frames: str,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio
        self.num_frames = num_frames