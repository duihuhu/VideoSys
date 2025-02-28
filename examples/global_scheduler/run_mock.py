import os
import time

ratios = [(2,2,6),(2,6,2),(6,2,2),(2,4,4),(4,2,4),(4,4,2),(1,3,6),(6,1,3),(3,6,1),(1,1,1)]
base_command = "python3 resolution_helper.py --ratio1 {x} --ratio2 {y} --ratio3 {z}"
for x, y, z in ratios:
    command = base_command.format(x = x, y = y, z = z)
    os.system(command)
    time.sleep(1)

recv_ratio = [0.25, 0.5, 0.75, 1]
types = [0, 1, 2, 3, 4]
base_command = "python3 mock_test.py --low {x} --middle {y} --high {z} --recv-ratio {a} --type {b} --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/mock_stream/{x}_{y}_{z}_{a}_{b}_"
for x, y, z in ratios:
    for rr in recv_ratio:
        for t in types:
            command = base_command.format(x = x, y = y, z = z, a = rr, b = t)
            os.system(command)
            time.sleep(5)