import os
import time

ratios = [(1,1,8),(2,2,6),(3,3,4),(1,8,1),(2,6,2),(3,4,3),(8,1,1),(6,2,2),(4,3,3),(2,4,4),(4,2,4),(4,4,2),(1,3,6),(6,1,3),(3,6,1),(1,1,1)]
base_command = "python3 mock_test.py --low {x} --middle {y} --high {z} --log-file-path ..._{x}_{y}_{z}"

for x, y, z in ratios:
    command = base_command.format(x = x, y = y, z = z)
    os.system(command)
    time.sleep(1)