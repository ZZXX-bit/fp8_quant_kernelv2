#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_EQ(x, y) TORCH_CHECK((x) == (y), #x " must equal to " #y)

__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
  unsigned mask = 0xffffffff;
  
  val = fmaxf(val, __shfl_xor_sync(mask, val, 16));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

// 将输入张量按组量化为FP8(e4m3)格式
template <typename SrcType, bool IS_COLUMN_MAJOR = false>
__global__ void fp8_block_quant_kernel(
    const SrcType* __restrict__ input,
    c10::Float8_e4m3fn* __restrict__ output_q,
    float* __restrict__ output_s,
    const int group_size,
    const int num_groups,
    const int groups_per_block,
    const float eps,
    const float fp8_min,
    const float fp8_max,
    const int scale_num_rows = 0,
    const int scale_stride = 0) {
  const int threads_per_group = 32;  // 使用一个warp处理一个组
  const int local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;

  const int block_group_id = blockIdx.x * groups_per_block;
  const int global_group_id = block_group_id + local_group_id;
  
  if (global_group_id >= num_groups) return;
  
  const int block_group_offset = global_group_id * group_size;

  float local_absmax = eps;

  const SrcType* group_input = input + block_group_offset;
  c10::Float8_e4m3fn* group_output = output_q + block_group_offset;
  float* scale_output;

  // 根据行主序或列主序计算scale输出的位置
  if constexpr (IS_COLUMN_MAJOR) {
    const int row_idx = global_group_id / scale_num_rows;
    const int col_idx = global_group_id % scale_num_rows;
    scale_output = output_s + (col_idx * scale_stride + row_idx);
  } else {
    scale_output = output_s + global_group_id;
  }

  // 查找组内最大绝对值
  for (int i = lane_id; i < group_size; i += threads_per_group) {
    float val = static_cast<float>(group_input[i]);
    float abs_val = fabsf(val);
    local_absmax = fmaxf(local_absmax, abs_val);
  }

  // 在warp内规约找到最大值
  local_absmax = GroupReduceMax(local_absmax, lane_id);

  // 计算缩放因子
  const float scale = local_absmax / fp8_max;

  // 存储缩放因子
  if (lane_id == 0) {
    *scale_output = scale;
  }

  // 量化数据
  for (int i = lane_id; i < group_size; i += threads_per_group) {
    float val = static_cast<float>(group_input[i]);
    float q_val = fminf(fmaxf(val / scale, fp8_min), fp8_max);
    group_output[i] = c10::Float8_e4m3fn(q_val);
  }
}

void fp8_block_quantize(
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int64_t group_size,
    double eps = 1e-5,
    double fp8_min = -448.0,
    double fp8_max = 448.0,
    bool is_column_major = false) {

  CHECK_INPUT(input);

  const int num_elements = input.numel();
  CHECK_EQ(num_elements % group_size, 0);
  const int num_groups = num_elements / group_size;
  
  // 检查output_s尺寸是否匹配，列主序情况下需要检查总元素数
  CHECK_EQ(output_s.numel(), num_groups);
  CHECK_EQ(output_q.numel(), num_elements);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int threads_per_group = 32;  // 每组使用一个warp
  const int max_threads_per_block = 256;
  int groups_per_block = max_threads_per_block / threads_per_group;
  groups_per_block = std::min(groups_per_block, num_groups);

  const int num_blocks = (num_groups + groups_per_block - 1) / groups_per_block;
  const int num_threads = groups_per_block * threads_per_group;

  // 准备列主序相关参数，列数
  int scale_num_rows = 0;
  // 列间距
  int scale_stride = 0;
  
  if (is_column_major) {
    if (output_s.dim() != 2) {
      TORCH_CHECK(false, "Column-major mode requires output_s to be 2D tensor");
    }
    scale_num_rows = output_s.size(1);
    scale_stride = output_s.stride(1);
  }

  #define LAUNCH_KERNEL(SrcType) \
    if (is_column_major) { \
      fp8_block_quant_kernel<SrcType, true><<<num_blocks, num_threads, 0, stream>>>( \
          static_cast<SrcType*>(input.data_ptr()), \
          static_cast<c10::Float8_e4m3fn*>(output_q.data_ptr()), \
          static_cast<float*>(output_s.data_ptr()), \
          group_size, \
          num_groups, \
          groups_per_block, \
          static_cast<float>(eps), \
          static_cast<float>(fp8_min), \
          static_cast<float>(fp8_max), \
          scale_num_rows, \
          scale_stride); \
    } else { \
      fp8_block_quant_kernel<SrcType, false><<<num_blocks, num_threads, 0, stream>>>( \
          static_cast<SrcType*>(input.data_ptr()), \
          static_cast<c10::Float8_e4m3fn*>(output_q.data_ptr()), \
          static_cast<float*>(output_s.data_ptr()), \
          group_size, \
          num_groups, \
          groups_per_block, \
          static_cast<float>(eps), \
          static_cast<float>(fp8_min), \
          static_cast<float>(fp8_max)); \
    }

  if (input.scalar_type() == at::ScalarType::Float) {
    LAUNCH_KERNEL(float);
  } else if (input.scalar_type() == at::ScalarType::Half) {
    LAUNCH_KERNEL(at::Half);
  } else if (input.scalar_type() == at::ScalarType::BFloat16) {
    LAUNCH_KERNEL(at::BFloat16);
  } else {
    TORCH_CHECK(false, "Unsupported input data type");
  }

  #undef LAUNCH_KERNEL

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(error));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_block_quantize", &fp8_block_quantize, "FP8 Block Quantization (CUDA)",
        py::arg("input"), py::arg("output_q"), py::arg("output_s"),
        py::arg("group_size"), py::arg("eps") = 1e-5,
        py::arg("fp8_min") = -448.0, py::arg("fp8_max") = 448.0,
        py::arg("is_column_major") = false);
} 