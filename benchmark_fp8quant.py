# SPDX-License-Identifier: Apache-2.0

import time
import argparse
import torch
import numpy as np
from fp8_ops import per_token_group_quant_fp8
from vllm_fp8_quant.vllm_fp8_quant_kernel import per_token_group_quant_fp8 as vllm_per_token_group_quant_fp8

@torch.inference_mode()
def main(seq_len: int,
         hidden_size: int,
         group_size: int,
         dtype: torch.dtype,
         seed: int = 0,
         do_profile: bool = False,
         test_vllm: bool = False,
         num_warmup_iters: int = 5,
         num_iters: int = 100) -> None:
    # 设置随机种子以确保结果可重复
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 创建测试数据，范围为[-1000, 1000]
    x = torch.rand(seq_len, hidden_size, device="cuda", dtype=dtype) * 2000 - 1000
    y = torch.rand(seq_len, hidden_size, device="cuda", dtype=dtype) * 2000 - 1000
    
    # 确保hidden_size可以被group_size整除
    if hidden_size % group_size != 0:
        raise ValueError(f"hidden_size ({hidden_size})必须能被group_size ({group_size})整除")
    
    print(f"测试数据形状: {x.shape}, 组大小: {group_size}")
    
    def run_cuda_benchmark(num_iters: int, profile: bool = False) -> float:
        # GPU实现测试
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        
        start_time = time.perf_counter()
        for _ in range(num_iters):
            x_gpu_quant, scales_gpu = per_token_group_quant_fp8(x, group_size=group_size)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        
        return (end_time - start_time) / num_iters
    
    def run_vllm_benchmark(num_iters: int, profile: bool = False) -> float:
        # vLLM实现测试
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        
        start_time = time.perf_counter()
        for _ in range(num_iters):
            x_vllm_quant, scales_vllm = vllm_per_token_group_quant_fp8(x, group_size=group_size)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        
        return (end_time - start_time) / num_iters
    
    # 预热
    print("GPU预热中...")
    run_cuda_benchmark(num_iters=num_warmup_iters, profile=False)
    
    # GPU基准测试
    if do_profile:
        gpu_latency = run_cuda_benchmark(num_iters=1, profile=True)
    else:
        gpu_latency = run_cuda_benchmark(num_iters=num_iters, profile=False)
    
    print(f"GPU运行时间: {gpu_latency * 1000000:.3f} μs")
    
    # 可选vLLM基准测试
    if test_vllm:
        print("vLLM测试中...")
        # 预热
        print("vLLM预热中...")
        run_vllm_benchmark(num_iters=num_warmup_iters, profile=False)
        # vLLM基准测试
        if do_profile:
            vllm_latency = run_vllm_benchmark(num_iters=1, profile=True)
        else:
            vllm_latency = run_vllm_benchmark(num_iters=num_iters, profile=False)
    
        print(f"vLLM运行时间: {vllm_latency * 1000000:.3f} μs")
        print(f"GPU加速比: {vllm_latency / gpu_latency:.2f}x")
    
    
    # 性能估算
    elements_per_second = seq_len * hidden_size / gpu_latency
    print(f"\n性能估算:")
    print(f"每秒处理元素数: {elements_per_second / 1e9:.2f} G元素/秒")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="测试FP8分块量化算子的性能")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dtype", type=str, choices=["float", "half", "bfloat16"], default="float")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true", help="启用CUDA Profiler")
    parser.add_argument("--test-vllm", type=bool, default=True, help="是否测试vLLM实现")
    parser.add_argument("--num-warmup-iters", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=1000)

    args = parser.parse_args()
    
    # 转换数据类型
    dtype_map = {
        "float": torch.float32,
        "half": torch.float16,
        "bfloat16": torch.bfloat16
    }
    
    main(seq_len=args.seq_len,
         hidden_size=args.hidden_size,
         group_size=args.group_size,
         dtype=dtype_map[args.dtype],
         seed=args.seed,
         do_profile=args.profile,
         test_vllm=args.test_vllm,
         num_warmup_iters=args.num_warmup_iters,
         num_iters=args.num_iters) 