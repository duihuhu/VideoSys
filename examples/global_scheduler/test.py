import requests
WORKER_URL = "http://127.0.0.1:8000/request_workers"  #GS服务器的地址 P

def post_http_request(request_id, worker_ids) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "request_id": request_id,
        "worker_ids": worker_ids,
    }
    response = requests.post(WORKER_URL, headers=headers, json=pload, stream=True)
    return response


if __name__ == "__main__":
    post_http_request("", [0,1])
    