#python3 async_server.py --port 8001 --rank 1 --dworld-size 2
# python3 client.py --rank 0 --world-size 2 --group-name g1 --op create --dport 8000
import argparse
from typing import AsyncGenerator
import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from videosys.core.async_engine import AsyncSched

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None


async def asyc_forward_request(request_dict, api_url):
    headers = {"User-Agent": "Test Client"}
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url=api_url, json=request_dict,
                                    headers=headers) as response:
                if response.status == 200:
                    delimiter=b"\0"
                    buffer = b''  # 用于缓存数据块中的部分消息
                    async for chunk in response.content.iter_any():
                        buffer += chunk  # 将新的数据块添加到缓冲区中
                        while delimiter in buffer:
                            index = buffer.index(delimiter)  # 查找分隔符在缓冲区中的位置
                            message = buffer[:index]  # 提取从缓冲区起始位置到分隔符位置的消息
                            yield message.strip()  # 返回提取的消息
                            buffer = buffer[index + len(delimiter):]  # 从缓冲区中移除已提取的消息和分隔符
                else:
                    print(f"Failed response for request {response.status}")
    except aiohttp.ClientError as e:
         print(f"Request {request_dict['request_id']} failed: {e}")
    except Exception as e:
         print(f"Unexpected error for request {request_dict['request_id']}: {e}")


@app.post("/recv_request")
async def recv_request(request: Request) -> Response:
    request_dict = await request.json()
    request_id = request_dict.pop("request_id")
    prompt = request_dict.pop("prompt")
    resolution = request_dict.pop("resolution")
    aspect_ratio = request_dict.pop("aspect_ratio")
    num_frames = request_dict.pop("num_frames")
    results_generator = await sched.generate(request_id = request_id, prompt = prompt, \
        resolution = resolution, aspect_ratio = aspect_ratio,num_frames = num_frames)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)

    args = parser.parse_args()
    sched = AsyncSched()
    sched.create_consumer()
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)

