"""
EXACT replication of trad_ptf.py with all the depth and comprehensiveness.
This module preserves every detail of your original traditional convolution implementation with ptflops.
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

def benchmark_convolution(input_shape, output_channels, kernel_size=(3, 3), stride=(2, 2)):
    """
    EXACT replication of the benchmark_convolution function from trad_ptf.py
    """
    class SimpleConvNet(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride):
            super(SimpleConvNet, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)

        def forward(self, x):
            return self.conv(x)

    model = SimpleConvNet(input_shape[0], output_channels, kernel_size, stride)
    model.eval()

    # Generate random input data
    input_data = torch.randn(1, *input_shape)

    # Warm-up run
    with torch.no_grad():
        model(input_data)

    # Benchmark
    latencies = []
    for _ in range(100):
        start_time = time.time()
        with torch.no_grad():
            model(input_data)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds

    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    # Calculate MACs and parameters using ptflops
    if PTFLOPS_AVAILABLE:
        macs, params = get_model_complexity_info(model, input_shape, as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        # Convert MACs to FLOPs
        flops = 2 * macs
    else:
        # Fallback calculation
        params = sum(p.numel() for p in model.parameters())
        # Simple FLOP estimation
        out_h = (input_shape[1] - kernel_size[0]) // stride[0] + 1
        out_w = (input_shape[2] - kernel_size[1]) // stride[1] + 1
        flops = 2 * input_shape[0] * kernel_size[0] * kernel_size[1] * output_channels * out_h * out_w
        macs = flops // 2

    print(f"Kernel Size: {kernel_size}")
    print(f"Input Shape: {input_shape}")
    print(f"Output Channels: {output_channels}")
    print(f"Parameters: {params / 1e3:.2f}K")
    print(f"MACs: {macs / 1e6:.2f}M")
    print(f"FLOPs: {flops / 1e6:.2f}M")
    print("Latency Statistics:")
    print(f"  Mean: {mean_latency:.2f}ms")
    print(f"  Std Dev: {std_latency:.2f}ms")
    print(f"  Min: {min_latency:.2f}ms")
    print(f"  Max: {max_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms")
    print(f"  P99: {p99_latency:.2f}ms")

def run_original_trad_ptf_benchmark():
    """
    EXACT replication of the original main execution from trad_ptf.py
    """
    # Example usage
    input_shape = (3, 224, 224)  # (channels, height, width)
    print("Benchmarking with output channels = 64 (Small Output):")
    benchmark_convolution(input_shape, 64)   # Small Output

    print("\nBenchmarking with output channels = 128 (Medium Output):")
    benchmark_convolution(input_shape, 128)  # Medium Output

    print("\nBenchmarking with output channels = 256 (Large Output):")
    benchmark_convolution(input_shape, 256)  # Large Output

# Make this module directly executable like the original
if __name__ == "__main__":
    run_original_trad_ptf_benchmark()
