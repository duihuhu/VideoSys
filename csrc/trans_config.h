#pragma once
#ifndef TRANS_CONFIG_H
#define TRANS_CONFIG_H

#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include "nccl.h"
#include <nlohmann/json.hpp>  // Include the JSON library
#include <iostream>
#include <cuda_runtime.h>
#include <tuple>

using json = nlohmann::json;

class TransManager {
public:
    TransManager(int rank, int local_rank, int nccl_local_rank);

    ~TransManager();
    std::vector<char> get_nccl_id(const std::string& dst_channel, const std::string& worker_type);
    // void create_comm(std::vector<char>& nccl_id ,const std::string& dst_channel, const std::string& worker_type);
    void dist_worker();

private:
    std::thread execute;

    int rank;
    int local_rank;
    int nccl_local_rank;
};
#endif // TRANS_CONFIG_H
