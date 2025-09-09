'''import torch
import os
import imageio

# 定义张量的形状
shape = (1, 51, 144, 256, 3)

# 随机生成一个张量，范围是0-255，数据类型是float32
# 然后转换为uint8并移动到CPU
video = torch.randint(0, 256, shape, dtype=torch.uint8, device='cpu')

output_path = os.path.join("/workspace/VideoSys/outputs", "save_test.mp4")
    
os.makedirs(os.path.dirname(output_path), exist_ok=True)
imageio.mimwrite(output_path, video[0], fps=24)'''

import time
from videosys import OpenSoraConfig, VideoSysEngine

config = OpenSoraConfig(num_gpus=8)
engine = VideoSysEngine(config)

prompt = "Sunset over the sea."
st = time.time()
video = engine.generate(prompt).video[0]
engine.save_video(video, f"./test_outputs/bs1.mp4")
ed = time.time()
print(f"Video generation time for bs1: {ed - st} seconds")

prompt2 = ["Sunset over the sea."] * 10
st2 = time.time()
video2 = engine.generate(prompt2).video[0]
engine.save_video(video2, f"./test_outputs/bs10.mp4")
ed2 = time.time()
print(f"Video generation time for bs10: {ed2 - st2} seconds")