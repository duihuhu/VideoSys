from videosys.models.open_sora.vae import OpenSoraVAE_V1_2
import torch
import numpy as np
import time
# # 从 .npy 文件中读取 NumPy 数组
# numpy_array = np.load('tensor.npy')

# # 将 NumPy 数组转换为 PyTorch Tensor
# tensor = torch.from_numpy(numpy_array)

# # 如果需要，将 Tensor 移动到 GPU
# samples = tensor.to('cuda:0')

samples = torch.load('tensor_396913.pt').cuda()
print("samples ", samples.shape())
tdtype=torch.bfloat16
vae = OpenSoraVAE_V1_2(
    from_pretrained="/home/jovyan/hcch/models/model/OpenSora-VAE-v1.2/models/snapshots/33d153e9b5a9f771a8a84f98bd3f46458a8ed0bf",
    micro_frame_size=17,
    micro_batch_size=4,
).to(device='cuda:0', dtype=tdtype)
print("alread load ")
samples = vae.decode(samples.to(tdtype), num_frames=51)
