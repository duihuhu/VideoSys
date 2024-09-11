import argparse
import json
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from videosys import OpenSoraConfig, VideoSysEngine

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None

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
    video = engine.generate(
        prompt=prompt,
        resolution="480p",
        aspect_ratio="9:16",
        num_frames="2s",
    ).video[0]
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
    parser.add_argument("--host", type=str, defauelt="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    
    config = OpenSoraConfig(num_sampling_steps=30, cfg_scale=7.0, num_gpus=1)
    engine = VideoSysEngine(config)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)

