#include "trans_config.h"

TransManager::TransManager(int rank, int local_rank, int nccl_local_rank): rank(rank), local_rank(local_rank), nccl_local_rank(nccl_local_rank){
    execute = std::thread(&TransManager::dist_worker, this);
    std::cout<<"TransManager " <<std::endl;
}

TransManager::~TransManager() {
    if (execute.joinable()) {
        execute.join();
    }
}
void TransManager::dist_worker() {
    while (true) {
        // if(!worker_task_queue.empty()) {
        //     auto worker_task = worker_task_queue.pop_front();
        //     TaskType task_type = worker_task.type;
        //     TransWorker* task_worker = nullptr;
        //     switch (task_type) {
        //         case TaskType::TRANSFER_SEND_BLOCKS:
        //             task_worker = send_trans_workers[worker_task.meta.channel];
        //             task_worker->add_tasks(worker_task);
        //             break;
        //         case TaskType::TRANSFER_RECV_BLOCKS:
        //             task_worker = recv_trans_workers[worker_task.meta.channel];
        //             task_worker->add_tasks(worker_task);
        //             break;
        //         default:
        //             throw std::runtime_error("invalid task_type.");
        //     }
        // }
    }
    return;
}

std::vector<char> TransManager::get_nccl_id(const std::string& dst_channel, const std::string& worker_type){
    ncclUniqueId uniqueId; 
    ncclGetUniqueId(&uniqueId);
    // if(worker_type=="dit"){
    //     if(send_trans_workers.find(dst_channel) == send_trans_workers.end()){
    //         TransWorker* task_worker = new TransWorker(rank, local_rank, nccl_local_rank, dst_channel);
    //         send_trans_workers[dst_channel] = task_worker;
    //     }
    // } else{
    //     if(recv_trans_workers.find(dst_channel) == recv_trans_workers.end()){
    //         TransWorker* task_worker = new TransWorker(rank, local_rank, nccl_local_rank, dst_channel);
    //         recv_trans_workers[dst_channel] = task_worker;
    //     }
    // }
    return std::vector<char>(uniqueId.internal, uniqueId.internal + sizeof(uniqueId.internal));
}

void TransManager::create_comm(std::vector<char>& nccl_id ,const std::string& dst_channel, const std::string& worker_type){
    // if(worker_type=="sender"){
    //     if(send_trans_workers.find(dst_channel) == send_trans_workers.end()){
    //         TransWorker* task_worker = new TransWorker(cache_size_per_block, gpu_cache, rank, local_rank, nccl_local_rank, dst_channel, tp, num_layer, cache_block_size, blocks_gpu_cache);
    //         send_trans_workers[dst_channel] = task_worker;
    //     }
    //     TransWorker* task_worker = send_trans_workers[dst_channel];
    //     task_worker->add_comm_task(nccl_id);
    // } else{
    //     if(recv_trans_workers.find(dst_channel) == recv_trans_workers.end()){
    //         TransWorker* task_worker = new TransWorker(cache_size_per_block, gpu_cache, rank, local_rank, nccl_local_rank, dst_channel, tp, num_layer, cache_block_size, blocks_gpu_cache);
    //         recv_trans_workers[dst_channel] = task_worker;
    //     }
    //     TransWorker* task_worker = recv_trans_workers[dst_channel];
    //     task_worker->add_comm_task(nccl_id);
    // }
    return;
}