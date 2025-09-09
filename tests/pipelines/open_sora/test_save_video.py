import torch
import os
import imageio

# 定义张量的形状
shape = (1, 51, 144, 256, 3)

# 随机生成一个张量，范围是0-255，数据类型是float32
# 然后转换为uint8并移动到CPU
video = torch.randint(0, 256, shape, dtype=torch.uint8, device='cpu')

output_path = os.path.join("/workspace/Videosys/outputs", "save_test.mp4")
    
os.makedirs(os.path.dirname(output_path), exist_ok=True)
imageio.mimwrite(output_path, video[0], fps=24)