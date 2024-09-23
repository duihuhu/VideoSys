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
      .def("deserialize", &TransferTaskMeta::deserialize);

  py::class_<TransferTask>(trans_ops, "TransferTask")
      .def(py::init<const TransferTaskMeta&, 
                    long long,
                    TaskType>(),
            py::arg("meta"),
            py::arg("video_addr"),
            py::arg("type"))
      .def_readwrite("meta", &TransferTask::meta)
      .def_readwrite("video_addr", &TransferTask::video_addr)
      .def_readwrite("type", &TransferTask::type)
      .def("serialize", &TransferTask::serialize)
      .def_static("deserialize", &TransferTask::deserialize);

  py::enum_<TaskType>(trans_ops, "TaskType")
      .value("TRANSFER_SEND", TaskType::TRANSFER_SEND)
      .value("TRANSFER_RECV", TaskType::TRANSFER_RECV)
      .export_values();
}