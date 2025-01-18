import random
import pickle
import argparse
import os

def main(request_num: int, ratio1: int, ratio2: int, ratio3: int, file_path: str) -> None:
    base_resolutions = ['144p', '240p', '360p']
    ratios = [ratio1, ratio2, ratio3]
    nums = [round(request_num * (ratios[i] / sum(ratios))) for i in range(len(ratios))]
    resolutions = []
    for i, num in enumerate(nums):
        for _ in range(num):
            resolutions.append(base_resolutions[i])
    random.shuffle(resolutions)
    with open(file_path, 'wb') as file:
        pickle.dump(resolutions, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type = str, default = ".pkl")
    parser.add_argument("--requests-num", type = int, default = 128)
    parser.add_argument("--ratio1", type = int, default = 1)
    parser.add_argument("--ratio2", type = int, default = 1)
    parser.add_argument("--ratio3", type = int, default = 1)
    args = parser.parse_args()

    random.seed(42)
    temp_path = "resolution_" + str(args.ratio1) + "_" + str(args.ratio2) + "_" + str(args.ratio3) + args.suffix
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), temp_path)

    main(request_num = args.requests_num, ratio1 = args.ratio1, ratio2 = args.ratio2, ratio3 = args.ratio3, file_path = file_path)