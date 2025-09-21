#python3 async_server.py --port 8000 --rank 0 --dworld-size 2

#python3 async_server.py --port 8001 --rank 2 --dworld-size 2

#修改rank
#python3 create_comm_test.py 

#in comm directory
# python3 client.py --rank 0 --world-size 2 --group-name g1 --op create --dport 8000


# dissagg
export CUDA_VISIBLE_DEVICES=0,1
python3 async_server.py --port 8000 --enable-separate --worker-type dit --num-gpus 2

export CUDA_VISIBLE_DEVICES=2
python3 async_server.py --port 8001 --enable-separate --worker-type vae --num-gpus 1

python3 create_comm_test.py 

python3 client

