#include "trans_config.h"

TransManager::TransManager(int rank, int local_rank, std::string worker_type): rank(rank), local_rank(local_rank),worker_type(worker_type){
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
        if(!worker_task_queue.empty()) {
            auto worker_task = worker_task_queue.pop_front();
            TaskType task_type = worker_task.type;
            TransWorker* task_worker = nullptr;
            switch (task_type) {
                case TaskType::TRANSFER_SEND:
                    std::cout<<"dist_worker " << "TRANSFER_SEND "<<std::endl;
                    task_worker = send_trans_workers[worker_task.meta.channel];
                    task_worker->add_tasks(worker_task);
                    break;
                case TaskType::TRANSFER_RECV:
                    std::cout<<"dist_worker " << "TRANSFER_RECV "<<std::endl;
                    task_worker = recv_trans_workers[worker_task.meta.channel];
                    task_worker->add_tasks(worker_task);
                    break;
                default:
                    throw std::runtime_error("invalid task_type.");
            }
        }
    }
    return;
}

std::vector<char> TransManager::get_nccl_id(const std::string& dst_channel, const std::string& worker_type){
    ncclUniqueId uniqueId; 
    ncclGetUniqueId(&uniqueId);
    if(worker_type=="dit"){
        if(send_trans_workers.find(dst_channel) == send_trans_workers.end()){
            TransWorker* task_worker = new TransWorker(rank, local_rank, dst_channel, worker_type);
            send_trans_workers[dst_channel] = task_worker;
        }
    } else{
        if(recv_trans_workers.find(dst_channel) == recv_trans_workers.end()){
            TransWorker* task_worker = new TransWorker(rank, local_rank, dst_channel, worker_type);
            recv_trans_workers[dst_channel] = task_worker;
        }
    }
    return std::vector<char>(uniqueId.internal, uniqueId.internal + sizeof(uniqueId.internal));
}

void TransManager::create_comm(std::vector<char>& nccl_id ,const std::string& dst_channel, const std::string& worker_type){
    if(worker_type=="dit"){
        if(send_trans_workers.find(dst_channel) == send_trans_workers.end()){
            TransWorker* task_worker = new TransWorker(rank, local_rank, dst_channel, worker_type);
            send_trans_workers[dst_channel] = task_worker;
        }
        TransWorker* task_worker = send_trans_workers[dst_channel];
        task_worker->add_comm_task(nccl_id);
    } else{
        if(recv_trans_workers.find(dst_channel) == recv_trans_workers.end()){
            TransWorker* task_worker = new TransWorker(rank, local_rank, dst_channel, worker_type);
            recv_trans_workers[dst_channel] = task_worker;
        }
        TransWorker* task_worker = recv_trans_workers[dst_channel];
        task_worker->add_comm_task(nccl_id);
    }
    return;
}

void TransManager::add_tasks(const std::vector<std::string>& tasks) {
    for (const auto& task : tasks) {
        auto trans_task = TransferTask::deserialize(task);
        worker_task_queue.push_back(trans_task);
    }
}

 std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>>TransManager::get_finished_transfer_tasks() {
     std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> finished_work_tasks;
    for (const auto& pair : send_trans_workers) {
        // const std::string& key = pair.first;
        TransWorker* worker = pair.second;
        auto finished_work_task = worker->get_finished_transfer_tasks();
        if(!finished_work_task.empty()) {
            finished_work_tasks.emplace_back(finished_work_task);
        }
    }
    for (const auto& pair : recv_trans_workers) {
        // const std::string& key = pair.first;
        TransWorker* worker = pair.second;
        auto finished_work_task = worker->get_finished_transfer_tasks();
        if(!finished_work_task.empty()) {
            finished_work_tasks.emplace_back(finished_work_task);
        }
    }
    return finished_work_tasks;
}