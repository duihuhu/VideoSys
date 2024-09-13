import json
from typing import Iterable, List

import requests
import uuid

G_URL = "http://127.0.0.1:8000/generate"  #DiT服务器的地址


def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    
    pload = {
        "prompt": "Sunset over the sea.",
        "request_id": random_uuid(), 
        "resolution": "480p",
        "aspect_ratio": "9:16",
        "num_frames": "2s",
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

def post_request_and_get_response(args, prompt):
    rsp = post_http_request(prompt, G_URL, args.n, args.stream)
    if args.stream:
        for h in get_streaming_response(rsp):
            # if h['finished'] == True:
            #     print("res", h)
            #     return h["prefilled_token_id"]
            print("response ", h )
                
def main(args, prompts):
    post_request_and_get_response(args, prompts)


if __name__ == "__main__":
    main()
    