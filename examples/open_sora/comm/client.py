import argparse
import json
from typing import Iterable, List
import requests
import uuid

G_URL = "http://127.0.0.1:8001/generate"  #GS服务器的地址 P


def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt, resolution, aspect_ratio, num_frames, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "request_id": random_uuid(),
        "prompt": prompt,
        "resolution": resolution, 
        "aspect_ratio": aspect_ratio,
        "num_frames": num_frames,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            # output = data["text"]
            yield data


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output

def post_request_and_get_response(prompt, resolution, aspect_ratio, num_frames):
    rsp = post_http_request(prompt, resolution, aspect_ratio, num_frames, G_URL)
    for h in get_streaming_response(rsp):
        if h['finished'] == True:
            print("res", h)
            return h["prefilled_token_id"]
            
def main(prompt, resolution, aspect_ratio, num_frames):
    post_request_and_get_response(prompt, resolution, aspect_ratio, num_frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Sunset over the sea.")
    parser.add_argument("--resolution", type=str, default="480p")
    parser.add_argument("--aspect-ratio", type=str, default="9:16")
    parser.add_argument("--num-frames", type=str, default="2s")
    args = parser.parse_args()
    
    prompt = args.prompt
    resolution = args.resolution
    aspect_ratio = args.aspect_ratio
    num_frames = args.num_frames
    
    main(prompt, resolution, aspect_ratio, num_frames)
    