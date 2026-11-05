// Shared (CPU + GPU) Metal memory as a torch MPS tensor, built by gguf/mps_shared.py.
//
// On Apple Silicon the CPU and GPU use the same RAM, but torch only hands out MPS tensors in GPU-private memory,
// so an expert read from the SSD has to be copied again to reach the GPU. This allocates the tensor from torch's
// own shared-memory MPS allocator instead (so every MPS op accepts it) and also returns the CPU address of its
// bytes, so os.preadv can read an expert straight into memory the GPU computes from.
#include <torch/extension.h>
#include <ATen/mps/MPSAllocatorInterface.h>

std::tuple<torch::Tensor, uint64_t> shared_empty(int64_t nbytes) {
  auto* alloc = at::mps::getIMPSAllocator(/*sharedAllocator=*/true);
  TORCH_CHECK(alloc->isSharedStorageSupported(), "this GPU has no memory shared with the CPU");
  c10::DataPtr data = alloc->allocate(nbytes);
  const void* cpu_address = alloc->getSharedBufferPtr(data.get()).first;
  TORCH_CHECK(cpu_address, "could not get the CPU address of a shared MPS buffer");
  c10::Storage storage(c10::Storage::use_byte_size_t(), nbytes, std::move(data), alloc, /*resizable=*/false);
  auto tensor = torch::empty({0}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kMPS));
  tensor.set_(storage, 0, {nbytes}, {1});
  return {tensor, reinterpret_cast<uint64_t>(cpu_address)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shared_empty", &shared_empty, "uint8 MPS tensor in shared memory, and the CPU address of its bytes");
}
