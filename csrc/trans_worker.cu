#include "trans_config.h"

TransWorker::TransWorker(int rank, int local_rank, const std::string& dst_channel, const std::string& worker_type):rank(rank), local_rank(local_rank), dst_channel(dst_channel), worker_type(worker_type) {
    trans_engine = TransEngine();
    if(worker_type=="vae"){
        comm_rank = 1;
    } else {
        comm_rank = 0;
    }
    use_comm = 0;
    execute = std::thread(&TransWorker::worker, this);
}


TransWorker::~TransWorker() {
    if (execute.joinable()) {
        execute.join();
    }
}

void TransWorker::init_device() {
    std::cout<<"before init device "<< local_rank<<std::endl;
    torch::Device device(torch::kCUDA, rank);
    c10::cuda::set_device(device.index());
    std::cout<<"after init device "<<std::endl;

}

void TransWorker::worker() {
    init_device();
    while (true) {
        if(!task_queue.empty()) {
            // std::cout<<"task_queue is not empty ";
            auto task = task_queue.pop_front();
            // TaskType task_type = task.type;
            auto task_meta = task.meta;
            switch (task_type) {
                case TaskType::TRANSFER_SEND:
                    // trans_engine.send_blocks(task_meta.channel, task_meta.request_id, task.blocks, dst_rank, comms[use_comm], streams[use_comm]);
                    // use_comm = (use_comm + 1) % comms.size();
                    break;
                case TaskType::TRANSFER_RECV:
                    // trans_engine.recv_blocks(task_meta.channel, task_meta.request_id, task.blocks, dst_rank, comms[use_comm], streams[use_comm]);
                    // use_comm = (use_comm + 1) % comms.size();
                    break;
                default:
                    throw std::runtime_error("invalid task_type.");
            }
        }

        auto send_blocks_finished = trans_engine.check_send_finished_events();
        auto recv_blocks_finished = trans_engine.check_recv_finished_events();
        
        if (!send_blocks_finished.empty() || !recv_blocks_finished.empty()){
            transfer_result_queue.push_back(std::make_pair(send_blocks_finished, recv_blocks_finished));
        }      
        
        while (!comm_queue.empty())
        {
            auto nccl_id = comm_queue.pop_front();
            ncclUniqueId uniqueId;
            std::memcpy(uniqueId.internal, nccl_id.data(), sizeof(uniqueId.internal));
            ncclComm_t comm = nullptr;
            if (trans_engine.create_nccl_comm(comm_rank, comm, uniqueId, 2)!=0) {
                throw std::runtime_error("CreateNcclFromRankTable error");
            }
            comms.push_back(comm);
            streams.push_back(c10::cuda::CUDAStream(c10::cuda::getStreamFromPool(true)));
        }
    }
}
void TransWorker::add_tasks(TransferTask& task) {
    task_queue.push_back(task);
}

void TransWorker::add_comm_task(std::vector<char>& nccl_id) {
    comm_queue.push_back(nccl_id);
}

std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>>TransWorker::get_finished_transfer_tasks() {
    std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> finished_tasks;
    while (!transfer_result_queue.empty())
    {
        // std::cout<<"transfer_result_queue is not empty ";
        auto finished_task = transfer_result_queue.pop_front();
        finished_tasks.push_back(finished_task);
    }
    return finished_tasks;
}