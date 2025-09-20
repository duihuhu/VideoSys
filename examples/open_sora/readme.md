#python3 async_server.py --port 8000 --rank 0 --dworld-size 2

#python3 async_server.py --port 8001 --rank 2 --dworld-size 2

#修改rank
#python3 create_comm_test.py 

#in comm directory
# python3 client.py --rank 0 --world-size 2 --group-name g1 --op create --dport 8000


python3 async_server.py --port 8000 --rank 0 --enable-separate --worker-type dit --gpus-num 1
python3 async_server.py --port 8001 --rank 0 --enable-separate --worker-type vae --gpus-num 1
#python3 create_comm_test.py 

