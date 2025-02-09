import argparse
import json
from typing import Iterable, List
import requests
import uuid
import time
import pickle
import numpy as np
import os
G_URL = "http://127.0.0.1:8001/recv_request"  #GS服务器的地址 P

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt, resolution, aspect_ratio, num_frames, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    request_id =  random_uuid()
    print(f"send request {request_id} resolution {resolution}")
    pload = {
        "request_id": request_id,
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
    # for h in get_streaming_response(rsp):
    #     print("res", time.time(), h)
    #print("rsp ", rsp)
            
def main(prompt, aspect_ratio, num_frames, res_path: str, recv_ratio: float, batch: bool):
    #t1 = time.time()
    add_resolutions = []
    with open(res_path, 'rb') as file:
        add_resolutions = pickle.load(file)

    if not batch:
        for resolution in add_resolutions:
            sleep_time = np.random.exponential(scale = 1 / recv_ratio, size = 1)[0]
            post_request_and_get_response(prompt, resolution, aspect_ratio, num_frames)
            time.sleep(sleep_time)
    else:
        add_resolutions = ['480p'] * 21
        for i, resolution in enumerate(add_resolutions):
            #if i == 32:
            #    break # add for debug
            post_request_and_get_response(prompt, resolution, aspect_ratio, num_frames)        
    #t2 = time.time()
    #print(t2-t1)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Sunset over the sea.")
    parser.add_argument("--aspect-ratio", type=str, default="9:16")
    parser.add_argument("--num-frames", type=str, default="2s")
    parser.add_argument("--ratio1", type = int, default = 1)
    parser.add_argument("--ratio2", type = int, default = 1)
    parser.add_argument("--ratio3", type = int, default = 1)
    parser.add_argument("--recv-ratio", type = float, default = 8.0)
    parser.add_argument("--batch", type = int, default = 1)
    args = parser.parse_args()
    
    np.random.seed(42)

    prompt = args.prompt
    aspect_ratio = args.aspect_ratio
    num_frames = args.num_frames
    
    temp_path = "resolution_" + str(args.ratio1) + "_" + str(args.ratio2) + "_" + str(args.ratio3) + ".pkl"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), temp_path)
    main(prompt, aspect_ratio, num_frames, file_path, args.recv_ratio, args.batch)