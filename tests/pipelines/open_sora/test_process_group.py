import itertools
import asyncio
import time


async def outer():
    await inner()
    
async def inner():
    print("before call A")
    return A()


async def A():
    print("before call B")
    await B()
    print("after call B")

async def B():
    print("hello")

async def call_api():
    print('Hello')
    await asyncio.sleep(3)
    print('World')

async def main():
    await outer()
    # start = time.perf_counter()
    # await asyncio.gather(call_api(), call_api())
    # end = time.perf_counter()
    # print(f'It took {end-start} second(s) to complete.')



def get_all_process_group(world_size: int):
    # generate permutation of all process groups 
    all_ranks = list(range(world_size))
    ranks_to_pg = {}
    parallel_sizes = [2 ** i for i in range(int.bit_length(world_size)) if 2 ** i <= world_size]
    for parallel_size in parallel_sizes:
        for pg_ranks in list(itertools.combinations(all_ranks, parallel_size)):
           ranks_to_pg[pg_ranks] = None
    print(ranks_to_pg)
    print(len(ranks_to_pg))
    for ranks in ranks_to_pg:
        ranks_to_pg[ranks] = sum(ranks)
    print(ranks_to_pg)
    print(len(ranks_to_pg))   




if __name__ == "__main__":
    # get_all_process_group(8)
    asyncio.run(main())
