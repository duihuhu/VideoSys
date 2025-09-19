import argparse
import json
from typing import Iterable, List
import requests
import uuid
import time
import pickle
import numpy as np
import os
import random
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
            
def main(prompt, aspect_ratio, num_frames, res_path: str, recv_ratio: float, batch: bool, sleep_path: str, tres: str):
    #t1 = time.time()
    #add_resolutions = []
    #with open(res_path, 'rb') as file:
    #    add_resolutions = pickle.load(file)
    
    #sleep_times = np.load(sleep_path)

    if not batch:
        #choices = ['360p', '720p']
        #weights = [12113, 4308]
        random.seed(42)
        #start_time = time.time()
        send = 0
        while send < 60:
            #if time.time() - start_time > 151:  # 30 seconds
            #    break
            #resolution = random.choices(choices, weights = weights, k = 1)[0]
            send += 1
            post_request_and_get_response(prompt, '360p', aspect_ratio, num_frames)
            time.sleep(1 / recv_ratio)
        #sleep_times = np.random.exponential(scale = 1 / recv_ratio, size = len(add_resolutions))
        #for j, resolution in enumerate(add_resolutions):
        #    post_request_and_get_response(prompt, resolution, aspect_ratio, num_frames)
        #    time.sleep(sleep_times[j])
        '''reqs_per_sample_30min = np.load("samples.npy")
        sleep_times_per_sample_30min = np.load("sleeps.npy")
        reqs_per_sample_5min = [reqs_per_sample_30min[0]]
        sleep_times_per_sample_5min = []
        reqs_st = 1
        sleeps_st = 0
        total_time = 0
        while sleeps_st < sleep_times_per_sample_30min.size:
            total_time += sleep_times_per_sample_30min[sleeps_st]
            if total_time > 30:
                break
            sleep_times_per_sample_5min.append(sleep_times_per_sample_30min[sleeps_st])
            reqs_per_sample_5min.append(reqs_per_sample_30min[reqs_st])
            sleeps_st += 1
            reqs_st += 1
        resolutions_5min = []
        ratio_360p = 12113
        ratio_720p = 4308
        total_reqs = sum(reqs_per_sample_5min)
        print(f"TOTAL REQS: {total_reqs}")
        res_360p = round(total_reqs * (ratio_360p / (ratio_360p + ratio_720p)))
        res_720p = total_reqs - res_360p
        for _ in range(res_360p):
            resolutions_5min.append('360p')
        for _ in range(res_720p):
            resolutions_5min.append('720p')
        random.shuffle(resolutions_5min)
        for i, reqs_num in enumerate(reqs_per_sample_5min):
            for _ in range(reqs_num):
                if len(resolutions_5min) == 0:
                    break
                resolution = resolutions_5min.pop()
                post_request_and_get_response(prompt, resolution, aspect_ratio, num_frames)
            if i < len(sleep_times_per_sample_5min):
                time.sleep(sleep_times_per_sample_5min[i])
        '''
    else:
        #for _ in range(1):
        #    post_request_and_get_response(prompt, tres, aspect_ratio, num_frames)
        add_resolutions = [tres] * 3
        for resolution in add_resolutions:
            post_request_and_get_response(prompt, resolution, aspect_ratio, num_frames)
        #add_resolutions = ['360p'] * 5
        #for i, resolution in enumerate(add_resolutions):
            #if i == 32:
            #    break # add for debug
        #    post_request_and_get_response(prompt, resolution, aspect_ratio, num_frames)
        #for temp in add_resolutions:
        #    for resolution in temp:
        #        post_request_and_get_response(prompt, resolution, aspect_ratio, num_frames)        
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
    parser.add_argument("--recv-ratio", type = float, default = 0.2)
    parser.add_argument("--batch", type = int, default = 1)
    parser.add_argument("--sleep", type = str, default = "")
    parser.add_argument("--tres", type = str, default = "144p")
    args = parser.parse_args()
    
    np.random.seed(42)

    prompt = args.prompt
    aspect_ratio = args.aspect_ratio
    num_frames = args.num_frames
    
    temp_path = "resolution_" + str(args.ratio1) + "_" + str(args.ratio2) + "_" + str(args.ratio3) + ".pkl"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), temp_path)
    main(prompt, aspect_ratio, num_frames, file_path, args.recv_ratio, args.batch, args.sleep, args.tres)