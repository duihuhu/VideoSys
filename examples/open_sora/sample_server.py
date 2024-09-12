import argparse
import json
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from videosys import OpenSoraConfig, VideoSysEngine
import time
import torch
from comm import CommData, CommEngine, CommonHeader, ReqMeta
TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None


async def query_hbm_meta(request_id, shape, vae_host, vae_port):
    vae_entry_point = (vae_host, vae_port)
    req_meta = ReqMeta(request_id, shape).__json__()
    data = CommData(
        headers=CommonHeader(vae_host, vae_port).__json__(),
        payload=req_meta
    )
    return await CommEngine.async_send_to(vae_entry_point, "allocate", data)

@app.post("/allocate")
async def allocate(request: Request) -> Response:
    request_dict = await request.json()
    print("request_dict ", request_dict) 
    video_shape = request_dict.pop("shape")
    t1 = time.time()
    samples = torch.empty(video_shape)
    t2 = time.time()
    print("allocate time ", t2-t1)
    ret = {"data_ptr": samples.data_ptr()}
    return JSONResponse(ret)


@app.post("/generate")
async def run_base(request: Request) -> Response:
    
    # request_dict = await request.json()
    # prompt = request_dict.pop("prompt")
    # prompt = request_dict.pop("prompt", "Sunset over the sea.")
    
    # change num_gpus for multi-gpu inference
    # sampling parameters are defined in the config
    # config = OpenSoraConfig(num_sampling_steps=30, cfg_scale=7.0, num_gpus=1)
    # engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    # num frames: 2s, 4s, 8s, 16s
    # resolution: 144p, 240p, 360p, 480p, 720p
    # aspect ratio: 9:16, 16:9, 3:4, 4:3, 1:1
    t1 = time.time()
    video = engine.generate(
        prompt=prompt,
        resolution="480p",
        aspect_ratio="9:16",
        num_frames="2s",
    ).video[0]
    t2 = time.time()
    print("execute time ", t2-t1)
    
    response = await query_hbm_meta("111", [1, 3, 51, 480, 848], "127.0.0.1", "8001")
    print("response ", response)
    engine.save_video(video, f"./outputs/{prompt}.mp4")
    """Health check."""
    return Response(status_code=200)

def run_pab():
    config = OpenSoraConfig(enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")

if __name__ == "__main__":
    # run_base()
    # run_pab()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num_gpus", type=int, default=1)

    args = parser.parse_args()
    
    config = OpenSoraConfig(num_sampling_steps=30, cfg_scale=7.0, num_gpus=args.num_gpus)
    engine = VideoSysEngine(config)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)

