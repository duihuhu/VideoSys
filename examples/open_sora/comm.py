import requests
from typing import Tuple, Dict
from enum import Enum
import aiohttp
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

class ReqMeta:
    def __init__(self, request_id, shape) -> None:
        self.request_id = request_id
        self.shape = shape
        
    def __json__(self) -> Dict:
        return {
            "request_id": self.request_id,
            "shape": self.shape,
        }
        
class CommData:
    def __init__(self, headers, payload) -> None:
        self.headers = headers
        self.payload = payload

class CommonHeader:
    def __init__(
        self,
        host: str,
        port: int,
    ) -> None:
        self.host = host
        self.port = port
    
    def __json__(self) -> Dict:
        return {
            "host": self.host,
            "port": str(self.port),
        }
        
class CommEngine:
    @staticmethod
    def send_to(entry_point: Tuple[str, int], func_name: str, data: CommData):
        api_url = f"http://{entry_point[0]}:{entry_point[1]}/{func_name}"
        response = requests.post(api_url, headers=data.headers, json=data.payload, stream=True)
        return response

    @staticmethod
    async def async_send_to(entry_point: Tuple[str, int], func_name: str, data: CommData):
        api_url = f"http://{entry_point[0]}:{entry_point[1]}/{func_name}"
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url=api_url, json=data.payload,
                                    headers=data.headers) as response:
                return await response.json()

