import argparse
import json
from typing import Iterable, List
import requests
import uuid
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def post_http_request(prompt, resolution, aspect_ratio, num_frames, role, request_id, api_url: str) -> requests.Response:
    print(f"send request {request_id} resolution {resolution} to {api_url}")
    headers = {"User-Agent": "Test Client"}
    pload = {
        "request_id": request_id,
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

def post_request_and_get_response(prompt, resolution, aspect_ratio, num_frames, role, GUR):
    _ = post_http_request(prompt, resolution, aspect_ratio, num_frames, role, random_uuid(), GUR)
    #for h in get_streaming_response(rsp):
    #    print("res", time.time(), h)
            
def main(prompt, aspect_ratio, num_frames, batch, recv_ratio):
    if not batch:
        random.seed(42)
        resolutions: List[str] = []
        GURs: List[str] = []
        roles: List[int] = []
        for _ in range(47):
            resolutions.append('360p')
        for _ in range(17):
            resolutions.append('720p')
        random.shuffle(resolutions)
        for i in range(64):
            if i % 2 == 0:
                GURs.append(G_URL1) 
                roles.append(0)
            else:
                GURs.append(G_URL2)
                roles.append(1)

        '''for resolution, GUR, role in zip(resolutions, GURs, roles):
            post_request_and_get_response(prompt, resolution, aspect_ratio, num_frames, role, GUR)
            time.sleep(1 / recv_ratio)
        '''

        # 构造任务
        tasks = list(zip(resolutions, GURs, roles))

        # 最大并发线程数 = recv_ratio
        max_workers = int(recv_ratio) if recv_ratio > 0 else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for resolution, GUR, role in tasks:
                futures.append(
                    executor.submit(
                        post_request_and_get_response,
                        prompt, resolution, aspect_ratio, num_frames, role, GUR
                    )
                )
                # 这里可以选择加上速率限制
                time.sleep(1.0 / recv_ratio)

            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    result = future.result()
                    # 如果需要处理返回结果，可以在这里加逻辑
                    # print(result)
                except Exception as e:
                    print(f"Task failed: {e}")
    else:
        add_resolutions = ['144p', '144p', '144p']
        for resolution in add_resolutions:
            post_request_and_get_response(prompt, resolution, aspect_ratio, num_frames, 0, G_URL1)

    
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
