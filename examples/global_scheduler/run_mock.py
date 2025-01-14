import os
import time

ratios = [(2,2,6),(2,6,2),(6,2,2),(2,4,4),(4,2,4),(4,4,2),(1,3,6),(6,1,3),(3,6,1),(1,1,1)]
base_command = "python3 mock_test.py --low {x} --middle {y} --high {z} --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity2/{x}_{y}_{z}_"

for x, y, z in ratios:
    command = base_command.format(x = x, y = y, z = z)
    os.system(command)
    time.sleep(1)
'''python3 mock_test.py --low 1 --middle 1 --high 8 --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity/1_1_8_
python3 mock_test.py --low 2 --middle 2 --high 6 --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity/2_2_6_
python3 mock_test.py --low 3 --middle 3 --high 4 --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity/3_3_4_
python3 mock_test.py --low 1 --middle 8 --high 1 --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity/1_8_1_
python3 mock_test.py --low 2 --middle 6 --high 2 --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity/2_6_2_
python3 mock_test.py --low 3 --middle 4 --high 3 --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity/3_4_3_
python3 mock_test.py --low 8 --middle 1 --high 1 --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity/8_1_1_
python3 mock_test.py --low 6 --middle 2 --high 2 --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity/6_2_2_
python3 mock_test.py --low 4 --middle 3 --high 3 --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity/4_3_3_
python3 mock_test.py --low 2 --middle 4 --high 4 --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity/2_4_4_
python3 mock_test.py --low 4 --middle 2 --high 4 --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity/4_2_4_
python3 mock_test.py --low 4 --middle 4 --high 2 --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity/4_4_2_
python3 mock_test.py --low 1 --middle 3 --high 6 --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity/1_3_6_
python3 mock_test.py --low 6 --middle 1 --high 3 --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity/6_1_3_
python3 mock_test.py --low 3 --middle 6 --high 1 --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity/3_6_1_
python3 mock_test.py --low 1 --middle 1 --high 1 --log-file-path /home/jovyan/hcch/hucc/VideoSys/examples/global_scheduler/batch_high_affinity/1_1_1'''