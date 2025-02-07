import os
import time

ratios = [(2,2,6),(2,6,2),(6,2,2),(2,4,4),(4,2,4),(4,4,2),(1,3,6),(6,1,3),(3,6,1),(1,1,1)]
policies = ["bandwidth", "unify", "group"]

base_command = "python3 policy_test.py --workload1-num {x} --workload2-num {y} --workload3-num {z} --log /home/jovyan/hhy/VideoSys/examples/global_scheduler/logs_{x}_{y}_{z}_"
base_command2 = "python3 policy_test.py --decouple --workload1-num {x} --workload2-num {y} --workload3-num {z} --log /home/jovyan/hhy/VideoSys/examples/global_scheduler/logs_{x}_{y}_{z}_"
base_command3 = "python3 bandwidth_aware_schedule.py --weight1 {x} --weight2 {y} --weight3 {z} --log /home/jovyan/hhy/VideoSys/examples/global_scheduler/logs_{x}_{y}_{z}.txt"
base_command4 = "python3 bandwidth_aware_schedule.py --weight1 {x} --weight2 {y} --weight3 {z} --log /home/jovyan/hhy/VideoSys/examples/global_scheduler/logs_{p}_{x}_{y}_{z}.txt --unify"
base_command5 = "python3 upper_bound.py --weight1 {x} --weight2 {y} --weight3 {z}"

for x, y, z in ratios:
    command = base_command5.format(x = x, y = y, z = z)
    '''for i, policy in enumerate(policies):
        if i == 0:
            command = base_command3.format(p = policy, x = x, y = y, z = z)
        elif i == 1:
            command = base_command4.format(p = policy, x = x, y = y, z = z)
        else:
            command = base_command5.format(p = policy, x = x, y = y, z = z)'''
    os.system(command)
    time.sleep(1)