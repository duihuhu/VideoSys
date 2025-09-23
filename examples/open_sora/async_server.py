#python3 async_server.py --port 8001 --rank 1 --dworld-size 2
# python3 client.py --rank 0 --world-size 2 --group-name g1 --op create --dport 8000
import argparse
import json
from typing import AsyncGenerator
import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from videosys import OpenSoraConfig
from videosys.core.async_engine import AsyncEngine
import time
import torch
from comm import CommData, CommEngine, CommonHeader, ReqMeta
from videosys.utils.config import DeployConfig
import videosys.entrypoints.server_config as cfg
import videosys.entrypoints.server_config2 as cfg2
from videosys.core.outputs import KvPreparedResponse
import os

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None

log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "start_log.txt")

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
        

async def query_hbm_meta(request_id, shape, vae_host, vae_port):
    vae_entry_point = (vae_host, vae_port)
    req_meta = ReqMeta(request_id, shape).__json__()
    data = CommData(
        headers=CommonHeader(vae_host, vae_port).__json__(),
        payload=req_meta
    )
    return await CommEngine.async_send_to(vae_entry_point, "kv_allocate", data)

@app.post("/kv_allocate")
async def kv_allocate(request: Request) -> Response:
    request_dict = await request.json()
    print("request_dict ", request_dict) 
    video_shape = request_dict.pop("shape")
    t1 = time.time()
    samples = torch.empty(video_shape)
    t2 = time.time()
    print("allocate time ", t2-t1)
    ret = {"data_ptr": samples.data_ptr()}
    return JSONResponse(ret)

    
@app.post("/get_nccl_id")
async def get_nccl_id(request: Request) -> Response:
    payload = await request.json()
    dst_channel = payload.pop("dst_channel")
    worker_type = payload.pop("worker_type")
    nccl_ids = await engine.get_nccl_id(dst_channel, worker_type)
    return nccl_ids

@app.post("/create_comm")
async def create_comm(request: Request) -> Response:
    payload = await request.json()
    dst_channel = payload.pop("dst_channel")
    nccl_id =  payload.pop("nccl_id")
    worker_type = payload.pop("worker_type")
    await engine.create_comm(nccl_id, dst_channel, worker_type)

@app.post("/generate_vae")
async def generate_vae(request: Request) -> Response:
    request_dict = await request.json()
    request_id = request_dict.pop("request_id")
    prompt = request_dict.pop("prompt")
    shape = request_dict.pop("shape")
    global_ranks = request_dict.pop("global_ranks")
    #print("generate_vae ", shape, global_ranks)
    results_generator = engine.generate(request_id=request_id, prompt=prompt, shape=shape,\
        global_ranks=global_ranks)
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for kv_response in results_generator:
            yield (json.dumps(kv_response.__json__()) + "\0").encode("utf-8")
    return StreamingResponse(stream_results()) 
   
@app.post("/generate_dit")
async def generate_dit(request: Request) -> Response:
    request_dict = await request.json()
    request_id = request_dict.pop("request_id")
    prompt = request_dict.pop("prompt")
    resolution = request_dict.pop("resolution")
    aspect_ratio = request_dict.pop("aspect_ratio")
    num_frames = request_dict.pop("num_frames")
    role = request_dict.pop("role")
    # request_id = "111"
    # prompt = "Sunset over the sea."
    # resolution = "480p"
    # aspect_ratio = "9:16"
    # num_frames = "2s"
    results_generator = engine.generate(request_id=request_id, prompt=prompt, resolution=resolution, \
        aspect_ratio=aspect_ratio, num_frames=num_frames)
    
    start_time = time.time()
    with open(log_path, 'a') as file:
        file.write(f"request {request_id} starts at {start_time}\n")
    
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            #print("request_output ", request_output)
            ret = {"text": "sucess "}
            if args.enable_separate:
                if role == 0:
                    dit_kv_resp = asyc_forward_request(request_output.__json__(), cfg.forward_vae_url % 
                                                        (cfg.vae_host, cfg.vae_port))
                else:
                    dit_kv_resp = asyc_forward_request(request_output.__json__(), cfg2.forward_vae_url % 
                                                        (cfg2.vae_host, cfg2.vae_port))
                async for resp in dit_kv_resp:
                    resp = resp.decode('utf-8')
                    #print("generate_dit resp ", resp)
                    payload = json.loads(resp)
                    global_ranks = payload.pop("global_ranks")
                    kv_response = KvPreparedResponse(**payload)
                    # print("response_kv_result ", kv_response.computed_blocks)
                    kv_response.global_ranks = global_ranks
                    await engine.add_kv_response(kv_response)
                    break
            yield (json.dumps(ret) + "\0").encode("utf-8")
    return StreamingResponse(stream_results())

@app.post("/create")
async def create(request: Request) -> Response:
    request_dict = await request.json()
    rank = request_dict.pop("rank")
    world_size = request_dict.pop("world_size")
    group_name = request_dict.pop("group_name")
    await engine.build_conn(rank=rank, world_size=world_size, group_name=group_name)
    
    ret = {'ret': 'success'}
    return JSONResponse(ret)


# @app.post("/generate")
# async def run_base(request: Request) -> Response:
    
#     # request_dict = await request.json()
#     # prompt = request_dict.pop("prompt")
#     # prompt = request_dict.pop("prompt", "Sunset over the sea.")
    
#     # change num_gpus for multi-gpu inference
#     # sampling parameters are defined in the config
#     # config = OpenSoraConfig(num_sampling_steps=30, cfg_scale=7.0, num_gpus=1)
#     # engine = VideoSysEngine(config)

#     prompt = "Sunset over the sea."
#     # num frames: 2s, 4s, 8s, 16s
#     # resolution: 144p, 240p, 360p, 480p, 720p
#     # aspect ratio: 9:16, 16:9, 3:4, 4:3, 1:1
#     t1 = time.time()
#     video = engine.generate(
#         prompt=prompt,
#         resolution="480p",
#         aspect_ratio="9:16",
#         num_frames="2s",
#     ).video[0]
#     t2 = time.time()
#     print("execute time ", t2-t1)
    
#     response = await query_hbm_meta("111", [1, 3, 51, 480, 848], "127.0.0.1", "8001")
#     print("response ", response)
#     engine.save_video(video, f"./outputs/{prompt}.mp4")
#     """Health check."""
#     return Response(status_code=200)

# def run_pab():
#     config = OpenSoraConfig(enable_pab=True)
#     engine = VideoSysEngine(config)

#     prompt = "Sunset over the sea."
#     video = engine.generate(prompt).video[0]
#     engine.save_video(video, f"./outputs/{prompt}.mp4")

if __name__ == "__main__":
    # run_base()
    # run_pab()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument('--enable-separate', action="store_true", help=('separate or not '))
    parser.add_argument('--worker-type', type=str, choices=['dit', 'vae'], default=None, help=('instance '))
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dworld-size', type=int, default=1)
    parser.add_argument('--vae-rank', type=int, default=0)

    args = parser.parse_args()
    
    deploy_config = DeployConfig()
    config = OpenSoraConfig(num_sampling_steps=30, cfg_scale=7.0, num_gpus=args.num_gpus, worker_type=args.worker_type, enable_separate=args.enable_separate, rank=args.rank, dworld_size = args.dworld_size)
    # engine = VideoSysEngine(config)
    engine = AsyncEngine(config, deploy_config, vae_rank = args.vae_rank)
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)

