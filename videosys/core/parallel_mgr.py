from typing import Optional

import torch
import torch.distributed as dist
from colossalai.cluster.process_group_mesh import ProcessGroupMesh
from torch.distributed import ProcessGroup

from videosys.utils.logging import init_dist_logger, logger
from videosys.utils.utils import set_seed

PARALLEL_MANAGER = None


class ParallelManager(ProcessGroupMesh):
    def __init__(self, parallel_group, dp_size, cp_size, sp_size):
        super().__init__(parallel_group, dp_size, cp_size, sp_size)
        dp_axis, cp_axis, sp_axis = 0, 1, 2

        self.dp_size = dp_size
        self.dp_group: ProcessGroup = self.get_group_along_axis(dp_axis)
        self.dp_rank = dist.get_rank(self.dp_group)
        # self.dp_rank = dist.get_rank(group=parallel_group)

        self.cp_size = cp_size
        self.cp_group: ProcessGroup = self.get_group_along_axis(cp_axis)
        self.cp_rank = dist.get_rank(self.cp_group)
        # self.cp_rank = dist.get_rank(group=parallel_group)

        self.sp_size = sp_size
        self.sp_group: ProcessGroup = self.get_group_along_axis(sp_axis)
        # print(f"new ß. global_rank: {torch.distributed.get_rank()}. local_sp_rank: {torch.distributed.get_rank(self.sp_group)}")
        self.sp_rank = dist.get_rank(self.sp_group)
        # self.sp_rank = dist.get_rank(group=parallel_group)
        
        self.enable_sp = sp_size > 1
        logger.info(f"Init parallel manager with dp_size: {dp_size}, cp_size: {cp_size}, sp_size: {sp_size} \n")


def set_parallel_manager(dp_size, cp_size, sp_size, parallel_group):
    global PARALLEL_MANAGER
    PARALLEL_MANAGER = ParallelManager(parallel_group, dp_size, cp_size, sp_size)

def del_parallel_manager():
    global PARALLEL_MANAGER
    PARALLEL_MANAGER = None

def get_data_parallel_group():
    return PARALLEL_MANAGER.dp_group


def get_data_parallel_size():
    return PARALLEL_MANAGER.dp_size


def get_data_parallel_rank():
    return PARALLEL_MANAGER.dp_rank


def get_sequence_parallel_group():
    return PARALLEL_MANAGER.sp_group


def get_sequence_parallel_size():
    return PARALLEL_MANAGER.sp_size


def get_sequence_parallel_rank():
    return PARALLEL_MANAGER.sp_rank


def get_cfg_parallel_group():
    return PARALLEL_MANAGER.cp_group


def get_cfg_parallel_size():
    return PARALLEL_MANAGER.cp_size


def enable_sequence_parallel():
    if PARALLEL_MANAGER is None:
        return False
    return PARALLEL_MANAGER.enable_sp


def get_parallel_manager():
    return PARALLEL_MANAGER

def initialize_device(local_rank=0, world_size= 0, distributed_init_method = None):
    torch.cuda.set_device(local_rank)
    initialize_position(local_rank, world_size, distributed_init_method)

def initialize_position(
    rank=0,
    # local_rank=0,
    world_size=1,
    init_method=None,
    # seed: Optional[int] = None,
    # sp_size: Optional[int] = None,
    # enable_cp: bool = False,
):
    if not dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        print("initialize_position method ")
        dist.init_process_group(backend="nccl", init_method=init_method, world_size=world_size, rank=rank, timeout=2)
        # torch.cuda.set_device(local_rank)
        init_dist_logger()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def initialize_manager(
    parallel_group,
    seed: Optional[int] = None,
    sp_size: Optional[int] = None,
    enable_cp: bool = False,
):

    # init sequence parallel
    if sp_size is None:
        sp_size = dist.get_world_size(group=parallel_group)
        dp_size = 1
    else:
        assert dist.get_world_size(group=parallel_group) % sp_size == 0, f"world_size {dist.get_world_size()} must be divisible by sp_size"
        dp_size = dist.get_world_size(group=parallel_group) // sp_size

    # update cfg parallel
    if enable_cp and sp_size % 2 == 0:
        sp_size = sp_size // 2
        cp_size = 2
    else:
        cp_size = 1
    set_parallel_manager(dp_size, cp_size, sp_size, parallel_group)

    if seed is not None:
        set_seed(seed + get_data_parallel_rank())


def send(self, z, rank):
    print("send data")
    dist.send(tensor=z, dst=1)
    
def recv(self, z, rank):
    print("recv data")

    dist.recv(tensor=z, src=0)
    
    
def destroy():
    if dist.is_initialized():
        dist.destroy_process_group()
        del_parallel_manager()

def initialize(
    rank=0,
    local_rank=0,
    world_size=1,
    init_method=None,
    seed: Optional[int] = None,
    sp_size: Optional[int] = None,
    enable_cp: bool = False,
):
    if not dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        print("init_method ", init_method)
        dist.init_process_group(backend="nccl", init_method=init_method, world_size=world_size, rank=rank)
        torch.cuda.set_device(local_rank)
        init_dist_logger()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # init sequence parallel
    if sp_size is None:
        sp_size = dist.get_world_size()
        dp_size = 1
    else:
        assert dist.get_world_size() % sp_size == 0, f"world_size {dist.get_world_size()} must be divisible by sp_size"
        dp_size = dist.get_world_size() // sp_size

    # update cfg parallel
    if enable_cp and sp_size % 2 == 0:
        sp_size = sp_size // 2
        cp_size = 2
    else:
        cp_size = 1

    set_parallel_manager(dp_size, cp_size, sp_size)

    if seed is not None:
        set_seed(seed + get_data_parallel_rank())

