from videosys import CogVideoXConfig, VideoSysEngine
import uuid
import argparse
#from videosys.utils.config import DeployConfig
#from typing import Dict, List
#import asyncio

def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def run_base(num_gpus: int = 1):#, height: int = 480):
    # models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
    # change num_gpus for multi-gpu inference
    config = CogVideoXConfig("/workspace/THUDM", num_gpus=num_gpus)

    #deploy_config = DeployConfig()
    #engine = VideoSysEngine(config, deploy_config=deploy_config)
    
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    # num frames should be <= 49. resolution is fixed to 720p.
    #width = height * 4 // 3

    #worker_ids: Dict[int, List[int]] = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3], 8: [0, 1, 2, 3, 4, 5, 6, 7]}
    #await engine.build_worker_comm(worker_ids=worker_ids.get(num_gpus, [0]))
    
    for _ in range(3):
        video = engine.generate(
            prompt=prompt,
            guidance_scale=6,
            num_inference_steps=50,
            num_frames=49,
            #height=height,
            #width=width,
        ).video[0]
    
    request_id = random_uuid()
    engine.save_video(video, f"/workspace/VideoSys/outputs/{request_id}.mp4")


def run_pab():
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_low_mem():
    config = CogVideoXConfig("THUDM/CogVideoX-2b", cpu_offload=True, vae_tiling=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--gpu", type=int, default=1, help="number of gpus to use for inference")
    #parser.add_argument("--height", type=int, default=720, help="height of the generated video")
    #args = parser.parse_args()
    for i in [2,4,8]:
        run_base(num_gpus=i)
    # run_pab()
    # run_low_mem()
