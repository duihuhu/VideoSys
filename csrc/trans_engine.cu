#include "trans_config.h"
#include <stdexcept>
#include <iostream>

void TransEngine::recv(const std::string& channel, const std::string& request_id, long long video_addr, int video_size, int opposite_rank, ncclComm_t& comm, c10::cuda::CUDAStream& stream) {

    c10::cuda::CUDAStreamGuard guard(stream);
    Recv(video_addr, video_size, opposite_rank, comm);

    at::cuda::CUDAEvent* event = new at::cuda::CUDAEvent();

    event->record();

    if (recv_events.find(channel) == recv_events.end()) {
        recv_events[channel] = std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>();
        recv_events[channel].push_back(std::make_pair(std::string(request_id), event));
    }
    else
        recv_events[channel].push_back(std::make_pair(request_id, event));
}

void TransEngine::send(const std::string& channel, const std::string& request_id, long long video_addr, int video_size, int opposite_rank, ncclComm_t& comm, c10::cuda::CUDAStream& stream) {

    c10::cuda::CUDAStreamGuard guard(stream);
    Send(video_addr, video_size, opposite_rank, comm);

    at::cuda::CUDAEvent* event = new at::cuda::CUDAEvent();
    event->record();
    if (send_events.find(channel) == send_events.end()) {
        send_events[channel] = std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>();
        send_events[channel].push_back(std::make_pair(request_id, event));
    } else
        send_events[channel].push_back(std::make_pair(request_id, event));
}


std::vector<std::string> TransEngine::check_send_finished_events() {
    std::vector<std::string> send_blocks_finished;
    for (auto& kv : send_events) {
        const std::string& channel = kv.first;
        auto& request_ids_and_events = kv.second;
        size_t num_finished_events = 0;

        for (auto it = request_ids_and_events.begin(); it != request_ids_and_events.end(); ++it) {
            const std::string& request_id = it->first;
            at::cuda::CUDAEvent *event = it->second;
            if (event->query()) {
                send_blocks_finished.emplace_back(TransferTaskMeta(channel, request_id).serialize());
                ++num_finished_events;
            } else {
                break;
            }
        }

        if (num_finished_events > 0) {
            // Remove finished events from the list
            request_ids_and_events.erase(request_ids_and_events.begin(), request_ids_and_events.begin() + num_finished_events);
        }
    }

    return send_blocks_finished;
}

std::vector<std::string> TransEngine::check_recv_finished_events() {
    std::vector<std::string> recv_blocks_finished;

    for (auto& kv : recv_events) {
        const std::string& channel = kv.first;
        auto& request_ids_and_events = kv.second;
        size_t num_finished_events = 0;

        for (auto it = request_ids_and_events.begin(); it != request_ids_and_events.end(); ++it) {
            const std::string& request_id = it->first;
            at::cuda::CUDAEvent *event = it->second;
            if (event->query()) {
                // std::cout<<"check_recv_finished_events " << request_id<< std::endl;
                recv_blocks_finished.emplace_back(TransferTaskMeta(channel, request_id).serialize());
                ++num_finished_events;
            } else {
                // std::cout<<"request_id not finished " << " " << request_id << std::endl;
                break;
            }
        }

        if (num_finished_events > 0) {
            // Remove finished events from the list
            request_ids_and_events.erase(request_ids_and_events.begin(), request_ids_and_events.begin() + num_finished_events);
        }
        
    }
    return recv_blocks_finished;
}


int TransEngine::create_nccl_comm(int32_t rank, ncclComm_t& comm, ncclUniqueId& uniqueId , int32_t NumDevice) {

    std::cout << "before create Global NCCL Comm " << rank << std::endl;
    ncclCommInitRank(&comm, NumDevice, uniqueId ,rank);
    std::cout << "Create Global NCCL Comm Success" << std::endl;
    return 0;
}


void TransEngine::Recv(long long video_addr, int video_size, uint32_t srcRank, ncclComm_t& comm)
{
    auto gpuStream = c10::cuda::getCurrentCUDAStream();
    auto cudaStream = gpuStream.stream();
    NCCLCHECK(ncclGroupStart());

    void *dstAddrPtr = (void*) video_addr;
    std::cout<<"Recv video size " << video_size << std::endl; 
    if (ncclSuccess != ncclRecv(dstAddrPtr, video_size, ncclInt8, srcRank,\
        comm, cudaStream)) {
        std::cout << "[ERROR]  ncclRecv error!!" << std::endl;
    }
    NCCLCHECK(ncclGroupEnd());
}


void TransEngine::Send(long long video_addr, int video_size, uint32_t destRank, ncclComm_t& comm)
{
    auto gpuStream = c10::cuda::getCurrentCUDAStream();
    auto cudaStream = gpuStream.stream();
    NCCLCHECK(ncclGroupStart());
    void *srcAddrPtr = (void*) video_addr;
    std::cout<<"Send video size " << video_size << std::endl; 

    if (ncclSuccess != ncclSend(srcAddrPtr, video_size, ncclInt8, destRank,\
        comm, cudaStream)) {
        std::cout << "[ERROR]  ncclSend error!!" << std::endl;
    }
    NCCLCHECK(ncclGroupEnd());
}

void TransEngine::checkNcclError(ncclResult_t result, const char* file, int line) {
    if (result != ncclSuccess) {
        std::cerr << "NCCL error at " << file << ":" << line << ": " 
                  << ncclGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}
