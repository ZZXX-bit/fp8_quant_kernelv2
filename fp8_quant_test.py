import torch
import numpy as np
import time
from fp8_ops import per_token_group_quant_fp8
from vllm_fp8_quant.vllm_fp8_quant_kernel import per_token_group_quant_fp8 as vllm_quant


def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建测试数据
    seq_len = 512
    hidden_size = 1024
    group_size = 128
    
    # 创建范围在[-1000, 1000]的随机数据
    x = torch.rand(seq_len, hidden_size, device='cuda') * 2000 - 1000
    
    print(f"开始测试FP8量化...")
    
    # GPU预热
    x_gpu_quant, scales_gpu = per_token_group_quant_fp8(x, group_size=group_size, column_major_scales=True)

    # 记录GPU实现的时间
    torch.cuda.synchronize()
    start = time.perf_counter()
    x_gpu_quant, scales_gpu = per_token_group_quant_fp8(x, group_size=group_size, column_major_scales=True)
    torch.cuda.synchronize()
    gpu_time = time.perf_counter() - start
    
    print(f"GPU实现完成，耗时: {gpu_time:.4f}秒")
    print(f"GPU量化结果形状: {x_gpu_quant.shape}, 缩放因子形状: {scales_gpu.shape}")

    # vLLM预热
    x_vllm_quant, scales_vllm = vllm_quant(x, group_size=group_size, column_major_scales=True)

    # 记录vllm实现的时间
    torch.cuda.synchronize()
    start = time.perf_counter()
    x_vllm_quant, scales_vllm = vllm_quant(x, group_size=group_size, column_major_scales=True)
    torch.cuda.synchronize()
    vllm_time = time.perf_counter() - start
    
    print(f"vllm实现完成，耗时: {vllm_time:.4f}秒")
    print(f"vllm量化结果形状: {x_vllm_quant.shape}, 缩放因子形状: {scales_vllm.shape}")
    
    
    # 将GPU结果转到CPU进行比较
    x_gpu_quant_cpu = x_gpu_quant.cpu().float()
    scales_gpu_cpu = scales_gpu.cpu()

    x_vllm_quant_cpu = x_vllm_quant.cpu().float()
    scales_vllm_cpu = scales_vllm.cpu()
    

    scales_diff = torch.abs(scales_gpu_cpu - scales_vllm_cpu).mean().item()
    
    quant_diff = torch.abs(x_gpu_quant_cpu - x_vllm_quant_cpu).mean().item()
    
    # 比较结果
    print("\n测试结果：")
    print(f"数据形状: {x.shape}, 组大小: {group_size}")
    print(f"GPU 执行时间: {gpu_time:.4f} 秒")
    print(f"vllm 执行时间: {vllm_time:.4f} 秒")
    print(f"加速比: {vllm_time / gpu_time:.2f}x")
    print(f"量化值平均差异: {quant_diff:.6f}")
    print(f"缩放因子平均差异: {scales_diff:.6f}")
    
    # 检查量化和反量化的一致性
    # 选择一些样本值进行详细比较
    sample_indices = [(0, 0), (255, 511), (100, 200)]
    print("\n样本值比较:")
    for idx in sample_indices:
        s, h = idx
        orig_val = x[s, h].item()
        gpu_val = x_gpu_quant[s, h].item()
        vllm_val = x_vllm_quant[s, h].item()
        
        group_idx = h // group_size
        
        # 确保可以安全获取缩放因子
        if s < scales_gpu.size(0) and group_idx < scales_gpu.size(1):
            gpu_scale = scales_gpu[s, group_idx].item()
        else:
            gpu_scale = float('nan')
            
        if s < scales_vllm.size(0) and group_idx < scales_vllm.size(1):
            vllm_scale = scales_vllm[s, group_idx].item()
        else:
            vllm_scale = float('nan')
        
        print(f"位置 {idx}:")
        print(f"  原始值: {orig_val:.4f}")
        print(f"  GPU量化值: {gpu_val:.4f} (缩放因子: {gpu_scale:.4f})")
        print(f"  vllm量化值: {vllm_val:.4f} (缩放因子: {vllm_scale:.4f})")
        print(f"  差异: {abs(gpu_val - vllm_val):.6f}")

if __name__ == "__main__":
    main()