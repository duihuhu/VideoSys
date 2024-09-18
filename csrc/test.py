from video_ops import trans_ops
trans_manager = trans_ops.TransManager(1,1,1)
nccl_id = trans_manager.get_nccl_id("1","222")
print("nccl id ", nccl_id)