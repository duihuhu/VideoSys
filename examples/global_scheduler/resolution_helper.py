import random
import pickle
import argparse
import os
import numpy as np

def main(request_num: int, ratio1: int, ratio2: int, ratio3: int, file_path: str, raw_data_path: str) -> None:
    base_resolutions = ['144p', '240p', '360p']
    ratios = [ratio1, ratio2, ratio3]
    #nums = [round(request_num * (ratios[i] / sum(ratios))) for i in range(len(ratios))]
    raw_data_nums = np.load(raw_data_path)
    raw_data_lengths = sum(raw_data_nums)
    raw_resolutions = []
    raw_lengths = [round(raw_data_lengths * (ratios[i] / sum(ratios))) for i in range(len(ratios))]
    for i, temp_resolution in enumerate(base_resolutions):
        for _ in range(raw_lengths[i]):
            raw_resolutions.append(temp_resolution)
    random.shuffle(raw_resolutions)
    resolutions = []
    st = 0
    for temp_nums in raw_data_nums:
        resolutions.append(raw_resolutions[st: st + temp_nums])
        st += temp_nums
    with open(file_path, 'wb') as file:
        pickle.dump(resolutions, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type = str, default = ".pkl")
    parser.add_argument("--requests-num", type = int, default = 128)
    parser.add_argument("--ratio1", type = int, default = 1)
    parser.add_argument("--ratio2", type = int, default = 1)
    parser.add_argument("--ratio3", type = int, default = 1)
    parser.add_argument("--raw", type = str, default = "")
    args = parser.parse_args()

    random.seed(42)
    temp_path = "resolution_" + str(args.ratio1) + "_" + str(args.ratio2) + "_" + str(args.ratio3) + args.suffix
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), temp_path)

    main(request_num = args.requests_num, ratio1 = args.ratio1, ratio2 = args.ratio2, ratio3 = args.ratio3, file_path = file_path, raw_data_path = args.raw)