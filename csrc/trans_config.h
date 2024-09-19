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
#include <nccl.h>
#include <nlohmann/json.hpp>  // Include the JSON library
#include <iostream>
#include <cuda_runtime.h>
#include <tuple>
#include "trans_queue.h"

using json = nlohmann::json;

// 定义TaskType枚举类型，用于区分不同的任务类型
enum class TaskType {
    TRANSFER_SEND,
    TRANSFER_RECV,
};

// TransferTaskMeta结构体，用于存储传输任务的元信息
class TransferTaskMeta {
public:
    TransferTaskMeta(){}

    TransferTaskMeta(const std::string& channel, const std::string& request_id)
        : channel(channel), request_id(request_id) {}
    std::string channel;
    std::string request_id;

    // Serialize TransferTaskMeta to JSON
    json to_json() const {
        return json{{"channel", channel}, {"request_id", request_id}};
    }

    // Serialize the TransferTask to a string (JSON format)
    std::string serialize() const {
        json task_meta;
        task_meta["channel"] = channel;
        task_meta["request_id"] = request_id;
        return task_meta.dump();
    }

    static TransferTaskMeta deserialize(const std::string& serialized_data) {
        json task_meta = json::parse(serialized_data); // Parse JSON string
        return TransferTaskMeta(task_meta.at("channel").get<std::string>(), task_meta.at("request_id").get<std::string>());
    }
    // Deserialize TransferTaskMeta from JSON
    static TransferTaskMeta from_json(const json& task_meta) {
        return TransferTaskMeta{task_meta.at("channel").get<std::string>(), task_meta.at("request_id").get<std::string>()};
    }
};


class TransferTask {
public:
    TransferTask(const TransferTaskMeta& meta, const std::vector<uint32_t>& blocks, TaskType type, int layer = 1, bool is_last_layer=false)
        : meta(meta), blocks(blocks), type(type), layer(layer), is_last_layer(is_last_layer) {}

    TransferTask(const TransferTaskMeta& meta, const std::vector<uint32_t>& blocks, const std::vector<uint32_t>& dst_blocks, TaskType type, int layer = 1, bool is_last_layer=false)
        : meta(meta), blocks(blocks), dst_blocks(dst_blocks), type(type), layer(layer), is_last_layer(is_last_layer) {}
    
    TransferTaskMeta meta;
    std::vector<uint32_t> blocks;
    std::vector<uint32_t> dst_blocks;
    TaskType type;
    int layer;
    bool is_last_layer;

    // Serialize the TransferTask to a string (JSON format)
    std::string serialize() const {
        json task;
        task["meta"] = meta.to_json();
        task["blocks"] = blocks;
        if (!dst_blocks.empty()) {
            task["dst_blocks"] = dst_blocks;
        }
        task["type"] = static_cast<int>(type);  // Store TaskType as an integer
        task["layer"] = layer;  // Store TaskType as an integer
        task["is_last_layer"] = is_last_layer;  // Store TaskType as an integer
        return task.dump();
    }

    // Deserialize from a string (JSON format) to a TransferTask object
    static TransferTask deserialize(const std::string& serialized_data) {
        json task = json::parse(serialized_data);
        TransferTaskMeta meta = TransferTaskMeta::from_json(task.at("meta"));
        std::vector<uint32_t> blocks = task.at("blocks").get<std::vector<uint32_t>>();
        std::vector<uint32_t> dst_blocks;
        if (task.contains("dst_blocks")) {
            dst_blocks = task.at("dst_blocks").get<std::vector<uint32_t>>();
        }
        TaskType type = static_cast<TaskType>(task.at("type").get<int>());
        int layer = task.at("layer").get<int>();
        bool is_last_layer = task.at("is_last_layer").get<bool>();

        if (dst_blocks.empty()) {
            return TransferTask(meta, blocks, type, layer, is_last_layer);
        } else {
            return TransferTask(meta, blocks, dst_blocks, type, layer, is_last_layer);
        }
    }
};

// TransEngine类，负责管理KV缓存并执行发送和接收操作
class TransEngine {
public:
    void recv_blocks(const std::string& channel, const std::string& request_id, const std::vector<uint32_t>& src_blocks, int opposite_rank, ncclComm_t& comm, c10::cuda::CUDAStream& stream);
    
    void send_blocks(const std::string& channel, const std::string& request_id,const std::vector<uint32_t>& dst_blocks, int opposite_rank, ncclComm_t& comm, c10::cuda::CUDAStream& stream);

    int create_nccl_comm(int32_t rank, ncclComm_t& comm, ncclUniqueId& uniqueId , int32_t NumDevice);

    std::vector<std::string> check_send_finished_events();
    std::vector<std::string> check_recv_finished_events();

    void SendBlocks(std::vector<std::pair<at::Tensor, at::Tensor>>& srcCaches, \
        const std::vector<uint32_t>& srcBlocks, uint32_t cacheSize, uint32_t destRank, ncclComm_t& comm);
    void RecvBlocks(std::vector<std::pair<at::Tensor, at::Tensor>>& dstCaches, \
        const std::vector<uint32_t>& dstBlocks, uint32_t cacheSize, uint32_t srcRank, ncclComm_t& comm);

    void checkNcclError(ncclResult_t result, const char* file, int line);
    // void throwError(const std::string& request_id, int blockIdx,  void* dstBlockPtr, int srcRank, size_t cacheSize);

private:

    // std::unordered_map<std::string, c10::cuda::CUDAStream*> send_streams;
    std::unordered_map<std::string, std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>> send_events;
    // std::unordered_map<std::string, c10::cuda::CUDAStream*> recv_streams;
    std::unordered_map<std::string, std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>> recv_events;
};

class TransWorker {
public:
    TransWorker(int rank, int local_rank, const std::string& dst_channel, std::string& worker_type);
    ~TransWorker();

    void add_tasks(TransferTask& task);
    std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> get_finished_transfer_tasks();

    void add_comm_task(std::vector<char>& uniqueId);
private:
    void init_device();
    void worker();

    TransEngine trans_engine;
    TransQueue<TransferTask> task_queue;
    TransQueue<std::vector<char>> comm_queue;
    TransQueue<std::pair<std::vector<std::string>, std::vector<std::string>>> transfer_result_queue;

    std::thread execute;
    int rank;
    int local_rank;
    std::string dst_channel;
    std::string worker_type;

    std::vector<int> dst_ranks;
    int comm_rank;
    int dst_rank;
    std::vector<ncclComm_t> comms;
    std::vector<c10::cuda::CUDAStream> streams;
    int use_comm;
};

class TransManager {
public:
    TransManager(int rank, int local_rank, std::string worker_type);

    ~TransManager();
    std::vector<char> get_nccl_id(const std::string& dst_channel, const std::string& worker_type);
    void create_comm(std::vector<char>& nccl_id ,const std::string& dst_channel, const std::string& worker_type);
    void dist_worker();

private:
    std::unordered_map<std::string, TransWorker*> send_trans_workers;
    std::unordered_map<std::string, TransWorker*> recv_trans_workers;

    std::thread execute;

    int rank;
    int local_rank;
    std::string worker_type;
};

#endif // TRANS_CONFIG_H
