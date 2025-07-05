"""
EXACT replication of DCNv1.py with all the depth and comprehensiveness.
This module preserves every detail of your original deformable convolution implementation.
"""
import math
import time
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.utils import _pair
from torchvision.ops import DeformConv2d

def calculate_parameters_and_flops(in_channels, out_channels, kernel_size, input_size, stride, padding, dilation, groups):
    """
    EXACT replication of the original calculate_parameters_and_flops function from DCNv1.py
    """
    weight_params = out_channels * (in_channels // groups) * (kernel_size ** 2)
    bias_params = out_channels
    total_params = weight_params + bias_params

    H_in = input_size
    W_in = input_size
    H_out = math.floor((H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    W_out = math.floor((W_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    flops_per_output = 2 * ((in_channels // groups) * (kernel_size ** 2))
    total_output_elements = H_out * W_out * out_channels
    total_flops = flops_per_output * total_output_elements

    return total_params, total_flops, (H_out, W_out, out_channels)

def benchmark_convolution(input_shape, output_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """
    EXACT replication of the original benchmark_convolution function from DCNv1.py
    """
    H, W, C = input_shape
    in_channels = C

    model = DeformConv2d(in_channels, output_channels, kernel_size, stride, padding, dilation, groups)
    model.eval()

    input_tensor = torch.randn(1, in_channels, H, W)

    _, _, output_shape = calculate_parameters_and_flops(in_channels, output_channels, kernel_size, H, stride, padding, dilation, groups)
    H_out, W_out, _ = output_shape

    offset_channels = 2 * (kernel_size ** 2) * (in_channels // groups)
    offset_tensor = torch.zeros(1, offset_channels, H_out, W_out)

    with torch.no_grad():
        model(input_tensor, offset_tensor)

    latencies = []
    for _ in range(100):
        start_time = time.time()
        with torch.no_grad():
            model(input_tensor, offset_tensor)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)

    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    total_params, total_flops, _ = calculate_parameters_and_flops(in_channels, output_channels, kernel_size, H, stride, padding, dilation, groups)

    print("Deformable Convolution Layer:")
    print(f"Input Shape: {input_tensor.shape}")
    print(f"Output Shape: {model(input_tensor, offset_tensor).shape}")
    print(f"Total Parameters: {total_params} ({total_params / 1e3:.2f}K)")
    print(f"Estimated FLOPs: {total_flops} (~{total_flops / 1e6:.2f}M)")
    print("Latency Statistics:")
    print(f"  Mean: {mean_latency:.2f}ms")
    print(f"  Std Dev: {std_latency:.2f}ms")
    print(f"  Min: {min_latency:.2f}ms")
    print(f"  Max: {max_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms")
    print(f"  P99: {p99_latency:.2f}ms")

def run_original_dcnv1_benchmark():
    """
    EXACT replication of the original main execution from DCNv1.py
    """
    # Example usage:
    input_shape = (224, 224, 3)

    print("Benchmarking with output channels = 64 (Small Output):")
    benchmark_convolution(input_shape, 64)

    print("\nBenchmarking with output channels = 128 (Medium Output):")
    benchmark_convolution(input_shape, 128)

    print("\nBenchmarking with output channels = 256 (Large Output):")
    benchmark_convolution(input_shape, 256)

# Make this module directly executable like the original
if __name__ == "__main__":
    run_original_dcnv1_benchmark()
