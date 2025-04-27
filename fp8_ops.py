import os
import torch
from torch.utils.cpp_extension import load

# 获取当前目录路径
current_dir = os.path.dirname(os.path.realpath(__file__))

# 查找环境变量中是否设置了TORCH_DIR
torch_dir = os.environ.get('TORCH_DIR', None)
torch_include_args = []
torch_library_args = []

if torch_dir:
    torch_include_args = [f"-I{os.path.join(torch_dir, 'include')}"]
    torch_library_args = [f"-L{os.path.join(torch_dir, 'lib')}"]

# 使用CUDA_HOME环境变量
cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
cuda_include_args = [f"-I{os.path.join(cuda_home, 'include')}"]

# 动态加载CUDA扩展
fp8_quant_cuda = load(
    name="fp8_quant_cudav1",
    sources=[
        os.path.join(current_dir, "fp8_quant_kernel.cu"),
    ],
    extra_include_paths=[os.path.join(cuda_home, 'include')],
    extra_cflags=torch_include_args + cuda_include_args + ["-O3"],
    extra_ldflags=torch_library_args,
    verbose=True
)

def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = None,
    column_major_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对输入张量执行按令牌组量化，将值量化为FP8表示。
    
    参数:
        x: 输入张量，ndim >= 2
        group_size: 用于量化的组大小
        eps: 避免除零的最小值
        dtype: 输出张量的数据类型，默认为平台支持的FP8类型
        column_major_scales: 是否使用列主序排列缩放因子
        
    返回:
        Tuple[torch.Tensor, torch.Tensor]: 量化张量（FP8类型）和量化缩放因子
    """
    # 确保输入在GPU上
    if not x.is_cuda:
        raise ValueError("输入张量必须在CUDA设备上")
    
    # 确保最后一个维度可以被group_size整除
    if x.size(-1) % group_size != 0:
        raise ValueError(f"输入张量的最后维度大小必须能被group_size整除，但 {x.size(-1)} 不能被 {group_size} 整除")
    
    # 创建输出张量
    # from vllm.platforms import current_platform
    # dtype = current_platform.fp8_dtype() if dtype is None else dtype
    dtype = torch.float8_e4m3fn if dtype is None else dtype
    
    # 创建输出张量
    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    
    
    # 根据column_major_scales决定scale的形状
    if column_major_scales:
        shape = (x.shape[-1] // group_size, ) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size, )
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    
    # 调用CUDA kernel
    fp8_info = torch.finfo(dtype)
    fp8_quant_cuda.fp8_block_quantize(
        x,
        x_q,
        x_s,
        group_size,
        eps=eps,
        fp8_min=fp8_info.min,
        fp8_max=fp8_info.max,
        is_column_major=column_major_scales
    )
    
    return x_q, x_s