import os
import time

ratios = [(2,2,6),(2,6,2),(6,2,2),(2,4,4),(4,2,4),(4,4,2),(1,3,6),(6,1,3),(3,6,1),(1,1,1)]
base_command = "python3 resolution_helper.py --ratio1 {x} --ratio2 {y} --ratio3 {z}"
for x, y, z in ratios:
    command = base_command.format(x = x, y = y, z = z)
    os.system(command)
    time.sleep(1)

'''base_command = "python3 mock_test.py --low {x} --middle {y} --high {z} --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity2/{x}_{y}_{z}_"
for x, y, z in ratios:
    command = base_command.format(x = x, y = y, z = z)
    os.system(command)
    time.sleep(1)'''