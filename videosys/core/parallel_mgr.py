from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from colossalai.cluster.process_group_mesh import ProcessGroupMesh
from torch.distributed import ProcessGroup
from functools import reduce
from operator import mul
from videosys.utils.logging import init_dist_logger, logger
from videosys.utils.utils import set_seed
import numpy as np

PARALLEL_MANAGER = None

def prod(nums: List[int]) -> int:
    """Product of a list of numbers.

    Args:
        nums (List[int]): A list of numbers.

    Returns:
        int: The product of the numbers.
    """
    return reduce(mul, nums)

class PgMesh():
    '''
    Inspired by colossalai.cluster.process_group_mesh, without creating process group. 
    '''    
    def __init__(self,world_size: int, rank: int,  *size: int) -> None:
        prod_size = prod(size)
        assert (
            prod_size == world_size
        ), f"The product of the size({prod_size}) must be equal to the world size({world_size})."

        self._shape = size
        self._rank = rank
        self._coord = PgMesh.unravel(self._rank, self._shape)
        self._ranks = []
        # self._ranks_to_group: Dict[Tuple[int, ...], ProcessGroup] = {}
        # self._group_to_ranks: Dict[ProcessGroup, Tuple[int, ...]] = {}    
    @staticmethod
    def ravel(coord: Tuple[int, ...], shape: Tuple[int, ...], mode: str = "raise") -> int:
        """Convert a coordinate to a rank.
           mode: ['raise', 'wrap', 'clip'], see https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html.
           with wrap, index out of range would be wrapped around.
           For instance, ravel((0, i, 0), (1, 2, 1), 'wrap') returns (i % 2)

        Args:
            coords (Tuple[int, ...]): Coordinate to be converted.
            shape (Tuple[int, ...]): Shape of the process group mesh.
            mode (Optional[str]): The mode for numpy.ravel_multi_index.

        Returns:
            int: Rank of the coordinate.
        """

        assert mode in ["raise", "wrap", "clip"]
        return int(np.ravel_multi_index(coord, shape, mode))
    @staticmethod
    def unravel(rank: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Convert a rank to a coordinate.

        Args:
            rank (int): Rank to be converted.
            shape (Tuple[int, ...]): Shape of the process group mesh.

        Returns:
            Tuple[int, ...]: Coordinate of the rank.
        """
        return np.unravel_index(rank, shape)
    @staticmethod
    def get_coords_along_axis(
        base_coord: Tuple[int, ...], axis: Union[int, List[int]], indices_at_axis: Union[List[int], List[List[int]]]
    ) -> List[Tuple[int, ...]]:
        """Get coordinates along the given axis.

        Args:
            base_coord (Tuple[int, ...]): Base coordinate which the coordinates along the axis are based on.
            axis (int): Axis along which the coordinates are generated.
            indices_at_axis (List[int]): Indices at the axis.

        Returns:
            List[Tuple[int, ...]]: Coordinates along the axis.
        """
        if isinstance(axis, int):
            axis = [
                axis,
            ]
            assert isinstance(indices_at_axis[0], int), f"Expected int, but got {type(indices_at_axis[0])}."
            indices_at_axis = [
                indices_at_axis,
            ]

        def add_index(base_coord, axis, indices_at_axis):
            coords_in_group = []
            for idx in indices_at_axis:
                coords_in_group.append(base_coord[:axis] + (idx,) + base_coord[axis + 1 :])
            return coords_in_group

        coords_in_group = [base_coord]
        for ax, indices_at_ax in zip(axis, indices_at_axis):
            new_coords_in_group = []
            for coords in coords_in_group:
                new_coords_in_group += add_index(coords, ax, indices_at_ax)
            coords_in_group = new_coords_in_group

        return coords_in_group
    def get_ranks_along_axis(self, axis: int,indices_at_axis: Optional[List[int]] = None):
        indices_at_axis = indices_at_axis or list(range(self._shape[axis]))
        coords_in_group = PgMesh.get_coords_along_axis(self._coord, axis, indices_at_axis)
        ranks_in_group = tuple([PgMesh.ravel(coord, self._shape) for coord in coords_in_group])
        return ranks_in_group
        # if ranks_in_group not in self._ranks_to_group:
        #     # no need to cache it explicitly, since it will be cached in `create_group_along_axis`
        #     return self.create_group_along_axis(axis, indices_at_axis, backend=backend)
        # return self._ranks_to_group[ranks_in_group]       

class ParallelManager:

    def __init__(
        self,
        dp_size,
        cp_size,
        sp_size,
        worker_ids: Tuple[int] = None,
        ranks_to_pg: Dict[Tuple[int, ...], ProcessGroup] = None,
    ):
        dp_axis, cp_axis, sp_axis = 0, 1, 2
        if (
            worker_ids == None
        ):  # use default process group. Generate other process group with ProcessGroupMesh
            pg_mesh = ProcessGroupMesh(dp_size, cp_size, sp_size)

            self.dp_size = dp_size
            self.dp_group: ProcessGroup = pg_mesh.get_group_along_axis(dp_axis)
            self.dp_rank = dist.get_rank(self.dp_group)

            self.cp_size = cp_size
            self.cp_group: ProcessGroup = pg_mesh.get_group_along_axis(cp_axis)
            self.cp_rank = dist.get_rank(self.cp_group)

            self.sp_size = sp_size
            self.sp_group: ProcessGroup = pg_mesh.get_group_along_axis(sp_axis)
            self.sp_rank = dist.get_rank(self.sp_group)

            self.enable_sp = sp_size > 1
            #logger.info(
            #    f"Init parallel manager with dp_size: {dp_size}, cp_size: {cp_size}, sp_size: {sp_size} \n"
            #)
            return
        # All possible process group are already generated. We just select suitable process groups
        global_rank = dist.get_rank()
        pg_mesh = PgMesh(
            len(worker_ids), worker_ids.index(global_rank), dp_size, cp_size, sp_size
        )

        self.dp_size = dp_size
        dp_local_ranks: List[int] = pg_mesh.get_ranks_along_axis(dp_axis)
        self.dp_ranks: Tuple[int] = tuple([worker_ids[idx] for idx in dp_local_ranks])
        self.dp_group = ranks_to_pg[self.dp_ranks]
        self.dp_rank = dist.get_rank(self.dp_group)

        self.cp_size = cp_size
        cp_local_ranks: List[int] = pg_mesh.get_ranks_along_axis(cp_axis)
        self.cp_ranks: Tuple[int] = tuple([worker_ids[idx] for idx in cp_local_ranks])
        self.cp_group = ranks_to_pg[self.cp_ranks]
        self.cp_rank = dist.get_rank(self.cp_group)

        self.sp_size = sp_size
        sp_local_ranks: List[int] = pg_mesh.get_ranks_along_axis(sp_axis)
        self.sp_ranks: Tuple[int] = tuple([worker_ids[idx] for idx in sp_local_ranks]) 
        self.sp_group = ranks_to_pg[self.sp_ranks]
        self.sp_rank = dist.get_rank(self.sp_group)

        self.enable_sp = sp_size > 1

        #print(f"[rank {global_rank}] ParallelMgr. dp_ranks: {self.dp_ranks}. cp_ranks: {self.cp_ranks}. sp_ranks: {self.sp_ranks}. dp_rank: {self.dp_rank}. cp_rank: {self.cp_rank}. sp_rank: {self.sp_rank}. worker_ids: {worker_ids}")
        #logger.info(
        #    f"Init parallel manager with dp_size: {dp_size}, cp_size: {cp_size}, sp_size: {sp_size} \n"
        #)       
        
        
        

    def get_ranks_along_axis(
        self, axis: int, indices_at_axis: Optional[List[int]] = None, backend: Optional[str] = None
    ) -> ProcessGroup:
        """Get the process group along the given axis which the current process belongs to. If the process group doesn't exist, it will be created.

        Args:
            axis (int): Axis along which the process groups are created.
            indices_at_axis (Optional[List[int]], optional): Indices at the axis. Defaults to None.
            backend (Optional[str], optional): Backend of the process group. Defaults to None.

        Returns:
            ProcessGroup: The process group along the given axis which the current process belongs to.
        """
        indices_at_axis = indices_at_axis or list(range(self._shape[axis]))
        coords_in_group = ProcessGroupMesh.get_coords_along_axis(self._coord, axis, indices_at_axis)
        ranks_in_group = tuple([ProcessGroupMesh.ravel(coord, self._shape) for coord in coords_in_group])
        if ranks_in_group not in self._ranks_to_group:
            # no need to cache it explicitly, since it will be cached in `create_group_along_axis`
            return self.create_group_along_axis(axis, indices_at_axis, backend=backend)
        return self._ranks_to_group[ranks_in_group]     


def set_parallel_manager(
    dp_size,
    cp_size,
    sp_size,
    worker_ids: List[int] = None,
    ranks_to_pg: Dict[Tuple[int, ...], ProcessGroup] = None,
):
    global PARALLEL_MANAGER
    PARALLEL_MANAGER = ParallelManager(
        dp_size, cp_size, sp_size, worker_ids, ranks_to_pg 
    )


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
        dist.init_process_group(backend="nccl", init_method=init_method, world_size=world_size, rank=rank)
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
    raise RuntimeError("function initialize_manager() is not used.")
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
    raise RuntimeError("function initialize() is not used.")
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
