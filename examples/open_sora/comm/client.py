import argparse
import json
from typing import Iterable, List
import requests
import uuid
import time
import random
G_URL1 = "http://127.0.0.1:8000/generate_dit"  #GS服务器的地址 P
G_URL2 = "http://127.0.0.1:8002/generate_dit"  #GS2服务器的地址 P2
ANS = 0

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt, resolution, aspect_ratio, num_frames, role, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "request_id": random_uuid(),
        "prompt": prompt,
        "resolution": resolution, 
        "aspect_ratio": aspect_ratio,
        "num_frames": num_frames,
        "role": role,
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
    global ANS
    if ANS % 2 == 0:
        G_URL = G_URL1
        role = 0
    else:
        G_URL = G_URL2
        role = 1
    ANS += 1
    print("post to ", G_URL)
    _ = post_http_request(prompt, resolution, aspect_ratio, num_frames, role, G_URL)
    #for h in get_streaming_response(rsp):
    #    print("res", time.time(), h)
            
def main(prompt, aspect_ratio, num_frames, batch, recv_ratio):
    if not batch:
        random.seed(42)
        resolutions: List[str] = []
        for _ in range(47):
            resolutions.append('360p')
        for _ in range(17):
            resolutions.append('720p')
        random.shuffle(resolutions)

        for resolution in resolutions:
            post_request_and_get_response(prompt, resolution, aspect_ratio, num_frames)
            time.sleep(1 / recv_ratio)
    else:
        add_resolutions = ['144p', '144p', '144p']
        for resolution in add_resolutions:
            post_request_and_get_response(prompt, resolution, aspect_ratio, num_frames)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Sunset over the sea.")
    parser.add_argument("--resolution", type=str, default="480p")
    parser.add_argument("--aspect-ratio", type=str, default="9:16")
    parser.add_argument("--num-frames", type=str, default="2s")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--recv-ratio", type=float, default=1.0)
    args = parser.parse_args()
    
    prompt = args.prompt
    resolution = args.resolution
    aspect_ratio = args.aspect_ratio
    num_frames = args.num_frames
    batch = args.batch
    recv_ratio = args.recv_ratio

    main(prompt, aspect_ratio, num_frames, batch, recv_ratio)
