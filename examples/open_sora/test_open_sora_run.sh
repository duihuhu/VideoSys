#!/bin/bash
echo "--world_size 8 --policy pab"
python3 sample.py --world_size 8 --policy pab
sleep 3

echo "--world_size 8 --policy pab"
python3 sample.py --world_size 6 --policy pab
sleep 3

echo "--world_size 8 --policy pab"
python3 sample.py --world_size 4 --policy pab
sleep 3

echo "--world_size 8 --policy pab"
python3 sample.py --world_size 2 --policy pab
sleep 3

echo "--world_size 8 --policy pab"
python3 sample.py --world_size 1 --policy pab
sleep 3

echo "--world_size 8 --policy base"
python3 sample.py --world_size 8 --policy base
sleep 3

echo "--world_size 8 --policy base"
python3 sample.py --world_size 6 --policy base
sleep 3

echo "--world_size 8 --policy base"
python3 sample.py --world_size 4 --policy base
sleep 3

echo "--world_size 8 --policy base"
python3 sample.py --world_size 2 --policy base
sleep 3

echo "--world_size 8 --policy base"
python3 sample.py --world_size 1 --policy base