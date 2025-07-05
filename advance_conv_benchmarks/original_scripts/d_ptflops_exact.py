"""
EXACT replication of D_Ptflops.py with all the depth and comprehensiveness.
This module preserves every detail of your original dynamic convolution implementation with ptflops.
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

class DynamicConv2D(nn.Module):
    """
    EXACT replication of the DynamicConv2D class from D_Ptflops.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DynamicConv2D, self).__init__()
        self.kernel_weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.kernel_weights, bias=self.bias, stride=self.stride, padding=self.padding)

def benchmark_convolution(input_shape, output_channels, kernel_size=3, stride=1, padding=1):
    """
    EXACT replication of the benchmark_convolution function from D_Ptflops.py
    """
    model = DynamicConv2D(input_shape[0], output_channels, kernel_size, stride, padding)
    model.eval()

    input_data = torch.randn(1, *input_shape)

    with torch.no_grad():
        model(input_data)

    latencies = []
    for _ in range(100):
        start_time = time.time()
        with torch.no_grad():
            model(input_data)
        latencies.append((time.time() - start_time) * 1000)

    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    if PTFLOPS_AVAILABLE:
        macs, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=True)
    else:
        # Fallback calculation
        params_count = sum(p.numel() for p in model.parameters())
        params = f"{params_count / 1e3:.2f}K" if params_count < 1e6 else f"{params_count / 1e6:.2f}M"
        
        # Simple FLOP estimation
        out_h = (input_shape[1] + 2*padding - kernel_size) // stride + 1
        out_w = (input_shape[2] + 2*padding - kernel_size) // stride + 1
        flops_count = 2 * input_shape[0] * kernel_size * kernel_size * output_channels * out_h * out_w
        macs = f"{flops_count / 2e6:.2f}M"

    print(f"Dynamic Convolution")
    print(f"Kernel Size: {kernel_size}x{kernel_size}")
    print(f"Input Shape: {input_shape}")
    print(f"Output Channels: {output_channels}")
    print(f"Parameters: {params}")
    print(f"MACs: {macs}")
    print("Latency Statistics:")
    print(f"  Mean: {mean_latency:.2f}ms")
    print(f"  Std Dev: {std_latency:.2f}ms")
    print(f"  Min: {min_latency:.2f}ms")
    print(f"  Max: {max_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms")
    print(f"  P99: {p99_latency:.2f}ms")
    print("\n")

def run_original_d_ptflops_benchmark():
    """
    EXACT replication of the original main execution from D_Ptflops.py
    """
    input_shape = (3, 224, 224)
    print("Benchmarking with output channels = 64 (Small Output):")
    benchmark_convolution(input_shape, 64)

    print("Benchmarking with output channels = 128 (Medium Output):")
    benchmark_convolution(input_shape, 128)

    print("Benchmarking with output channels = 256 (Large Output):")
    benchmark_convolution(input_shape, 256)

# Make this module directly executable like the original
if __name__ == "__main__":
    run_original_d_ptflops_benchmark()
