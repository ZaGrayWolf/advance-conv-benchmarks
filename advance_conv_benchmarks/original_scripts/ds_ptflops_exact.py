"""
EXACT replication of DS_Ptflops.py with all the depth and comprehensiveness.
This module preserves every detail of your original depthwise separable implementation with ptflops.
"""
import torch
import torch.nn as nn
import time
import numpy as np

# Optional ptflops import with graceful fallback
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False
    print("Warning: ptflops not available. Install with: pip install ptflops")

class DepthwiseSeparableConv(nn.Module):
    """
    EXACT replication of the DepthwiseSeparableConv class from DS_Ptflops.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

def benchmark_depthwise_separable_convolution(input_shape, output_channels, kernel_size=(3, 3), stride=(2, 2)):
    """
    EXACT replication of the benchmark_depthwise_separable_convolution function from DS_Ptflops.py
    """
    in_channels = input_shape[2]
    model = DepthwiseSeparableConv(in_channels, output_channels, kernel_size, stride)
    model.eval()

    input_data = torch.randn(1, in_channels, input_shape[0], input_shape[1])

    # Warm-up run for any lazy initialization
    with torch.no_grad():
        model(input_data)
    
    latencies = []
    for _ in range(100):
        start_time = time.time()
        with torch.no_grad():
            model(input_data)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # milliseconds

    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    # Calculate FLOPs and parameters using ptflops
    if PTFLOPS_AVAILABLE:
        macs, params = get_model_complexity_info(model, (in_channels, input_shape[0], input_shape[1]), as_strings=False, print_per_layer_stat=False)
        flops = macs * 2  # Convert MACs to FLOPs
    else:
        # Fallback calculation
        params = sum(p.numel() for p in model.parameters())
        # Simple FLOP estimation
        out_height = (input_shape[0] - kernel_size[0]) // stride[0] + 1
        out_width = (input_shape[1] - kernel_size[1]) // stride[1] + 1
        dw_flops = kernel_size[0] * kernel_size[1] * out_height * out_width * in_channels
        pw_flops = in_channels * output_channels * out_height * out_width
        flops = dw_flops + pw_flops
        macs = flops // 2

    # Compute output shape
    out_height = (input_shape[0] - kernel_size[0]) // stride[0] + 1
    out_width = (input_shape[1] - kernel_size[1]) // stride[1] + 1
    output_shape = (out_height, out_width, output_channels)
    
    print(f"Kernel Size: {kernel_size}")
    print(f"Input Shape: {input_shape}")
    print(f"Output Shape: {output_shape}")
    print(f"Parameters: {params / 1e3:.2f}K")
    print(f"Total FLOPs: {flops / 1e6:.2f}M")
    print("Latency Statistics:")
    print(f"  Mean: {mean_latency:.2f}ms")
    print(f"  Std Dev: {std_latency:.2f}ms")
    print(f"  Min: {min_latency:.2f}ms")
    print(f"  Max: {max_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms")
    print(f"  P99: {p99_latency:.2f}ms")

def run_original_ds_ptflops_benchmark():
    """
    EXACT replication of the original main execution from DS_Ptflops.py
    """
    # Example usage
    input_shape = (224, 224, 3) 

    print("Benchmarking with output channels = 64 (Small Output):")
    benchmark_depthwise_separable_convolution(input_shape, 64)

    print("\nBenchmarking with output channels = 128 (Medium Output):")
    benchmark_depthwise_separable_convolution(input_shape, 128)

    print("\nBenchmarking with output channels = 256 (Large Output):")
    benchmark_depthwise_separable_convolution(input_shape, 256)

# Make this module directly executable like the original
if __name__ == "__main__":
    run_original_ds_ptflops_benchmark()
