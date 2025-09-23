from typing import Dict, Set, List, Optional
import requests
import time
def post_request(api_url, request_dict: Optional[Dict] = {}):
    headers = {"User-Agent": "Test Client"}
    resp = requests.post(api_url, headers=headers, json=request_dict)
    return resp


def create_comm(dit_port, dit_rank, vae_port, vae_rank, worker_type):
    comm_uniqe_id_url = "http://%s:%s/get_nccl_id"
    create_comm_url = "http://%s:%s/create_comm"
    dit_host = "127.0.0.1"
    uniqe_id_api_url = comm_uniqe_id_url % (dit_host, dit_port)
    dst_channel = "_".join([str(rank) for rank in vae_rank])
    resp = post_request(uniqe_id_api_url, {"dst_channel": dst_channel, "worker_type":worker_type})
    
    creat_comm_api_url = create_comm_url % (dit_host, vae_port)
    src_channel =  "_".join([str(rank) for rank in dit_rank])
    payload = {}
    payload['nccl_id'] = resp.json()
    payload['dst_channel'] = src_channel
    payload['worker_type'] = "vae"
    print("payload ", payload)
    resp = post_request(creat_comm_api_url, payload)
    return resp

resp = create_comm(8000,[0],8001,[1], "dit")
resp2 = create_comm(8002,[0],8003,[1], "dit")
