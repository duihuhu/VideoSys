#include "trans_config.h"
#include <torch/extension.h>


PYBIND11_MODULE(video_ops, m) {
  pybind11::module trans_ops = m.def_submodule("trans_ops", "video gpu nccl utils");

  py::class_<TransManager>(trans_ops, "TransManager")
      .def(py::init<int, std::vector<std::pair<at::Tensor, at::Tensor>>&, int, int , int, int, int, int, std::vector<uint64_t>& >())
      .def("get_nccl_id", &TransManager::get_nccl_id, "A function that returns NCCL unique ID as a list of characters")
      .def("create_comm", &TransManager::create_comm, "A function create comm")
}
