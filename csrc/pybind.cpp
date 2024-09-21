#include "trans_config.h"
#include <torch/extension.h>


PYBIND11_MODULE(video_ops, m) {
  pybind11::module trans_ops = m.def_submodule("trans_ops", "video gpu nccl utils");

  py::class_<TransManager>(trans_ops, "TransManager")
      .def(py::init<int , int, std::string>())
      .def("get_nccl_id", &TransManager::get_nccl_id, "A function that returns NCCL unique ID as a list of characters")
      .def("create_comm", &TransManager::create_comm, "A function create comm");


  py::class_<TransferTaskMeta>(trans_ops, "TransferTaskMeta")
      .def(py::init<>())
      .def(py::init<const std::string&, const std::string& >())
      .def_readwrite("channel", &TransferTaskMeta::channel)
      .def_readwrite("request_id", &TransferTaskMeta::request_id)
      .def("serialize", &TransferTaskMeta::serialize)
      .def("deserialize", &TransferTaskMeta::deserialize);;

  py::class_<TransferTask>(trans_ops, "TransferTask")
      .def(py::init<const TransferTaskMeta&, 
                    const std::vector<uint32_t>&>())
      .def_readwrite("meta", &TransferTask::meta)
      .def_readwrite("blocks", &TransferTask::blocks)
      .def("serialize", &TransferTask::serialize)
      .def_static("deserialize", &TransferTask::deserialize);
}