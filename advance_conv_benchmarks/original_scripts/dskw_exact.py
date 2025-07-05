"""
EXACT replication of DSKW.py with all the depth and comprehensiveness.
This module preserves every detail of your original implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

class KWDSConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, num_kernels=4):
        """
        Dynamic Depthwise Separable Convolution (Kernel Warehouse version).
        Generates a dynamic depthwise kernel for each input channel via a kernel bank
        and an attention mechanism, then applies a fixed pointwise convolution.
        """
        super(KWDSConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.num_kernels = num_kernels
        self.out_channels = out_channels
        self.in_channels = in_channels
        
        # Kernel bank for depthwise convolution:
        # For depthwise, each input channel has its own kernel (shape: 1 per channel)
        # So the kernel bank shape: [num_kernels, in_channels, 1, kernel_size, kernel_size]
        self.depthwise_kernel_bank = nn.Parameter(
            torch.randn(num_kernels, in_channels, 1, kernel_size, kernel_size)
        )
        
        # Attention mechanism for kernel selection:
        # Takes in a global statistic (from in_channels) and outputs weights for each kernel candidate.
        self.attention_fc = nn.Linear(in_channels, num_kernels)
        self.softmax = nn.Softmax(dim=1)
        
        # Pointwise convolution to mix channels after depthwise convolution.
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        batch, channels, height, width = x.size()
        
        # Global Average Pooling to obtain channel-wise statistics.
        pooled = torch.mean(x, dim=[2, 3])  # shape: [B, in_channels]
        
        # Attention branch: produce weights for each kernel candidate.
        attn = self.attention_fc(pooled)      # shape: [B, num_kernels]
        attn = self.softmax(attn).view(batch, self.num_kernels, 1, 1, 1, 1)  # shape: [B, num_kernels, 1, 1, 1, 1]
        
        # Compute dynamic depthwise kernel.
        # depthwise_kernel_bank: [num_kernels, in_channels, 1, k, k] -> unsqueeze to [1, num_kernels, in_channels, 1, k, k]
        # Multiply by attention weights and sum over the num_kernels dimension.
        dynamic_depthwise_kernel = (self.depthwise_kernel_bank.unsqueeze(0) * attn).sum(dim=1)
        # Now dynamic_depthwise_kernel: [B, in_channels, 1, k, k]
        
        # For batch=1, remove extra batch dim so that weight shape matches conv2d expectations.
        if batch == 1:
            dynamic_depthwise_kernel = dynamic_depthwise_kernel.squeeze(0)  # becomes [in_channels, 1, k, k]
        
        # Apply depthwise convolution using the dynamic kernel.
        # Use groups = in_channels for depthwise.
        stride_val = (int(self.stride[0]), int(self.stride[1]))
        depthwise_out = F.conv2d(
            x, dynamic_depthwise_kernel, stride=stride_val, padding=self.kernel_size // 2, groups=self.in_channels
        )
        
        # Apply fixed pointwise convolution.
        out = self.pointwise_conv(depthwise_out)
        return out

def calculate_flops_and_params(input_shape, out_channels, kernel_size, stride, num_kernels=4):
    """
    EXACT replication of the original calculate_flops_and_params function from DSKW.py
    """
    in_channels = input_shape[2]
    # Output spatial dimensions (assuming padding = kernel_size//2)
    out_height = (input_shape[0] - kernel_size) // stride + 1
    out_width = (input_shape[1] - kernel_size) // stride + 1

    # -------------------------
    # Depthwise Branch
    # -------------------------
    # Depthwise kernel bank parameters:
    # Each candidate has: (kernel_size^2 * 1 * in_channels) parameters (ignoring bias for simplicity)
    # Total depthwise params = num_kernels * in_channels * kernel_size^2.
    dw_params = num_kernels * in_channels * (kernel_size * kernel_size)
    
    # Pointwise conv parameters:
    # (in_channels * out_channels) + out_channels (bias)
    pw_params = (in_channels * out_channels) + out_channels
    
    # Dynamic branch parameters (attention):
    # FC: in_channels -> num_kernels: (in_channels * num_kernels) + num_kernels
    attn_params = (in_channels * num_kernels) + num_kernels

    total_params = dw_params + pw_params + attn_params

    # -------------------------
    # FLOPs Estimation:
    # Depthwise conv FLOPs:
    # For each output element, depthwise multiplications: (kernel_size^2 * in_channels) operations over all channels.
    dw_mults = (kernel_size * kernel_size * in_channels) * out_height * out_width
    # Pointwise conv FLOPs:
    pw_mults = (in_channels * out_channels) * out_height * out_width
    # Dynamic branch FLOPs:
    # GAP: in_channels * (H*W) (roughly)
    gap_flops = in_channels * (input_shape[0] * input_shape[1])
    # FC layer: mapping in_channels -> num_kernels ~ 2 * in_channels * num_kernels
    fc_flops = 2 * in_channels * num_kernels
    # Softmax: assume ~2 * num_kernels
    softmax_flops = 2 * num_kernels
    # Kernel weighting: For each element in the depthwise kernel bank
    kernel_elements = in_channels * (kernel_size * kernel_size)
    weighting_flops = (num_kernels + (num_kernels - 1)) * kernel_elements

    dynamic_flops = gap_flops + fc_flops + softmax_flops + weighting_flops

    # Total multiplications:
    total_mults = dw_mults + pw_mults + dynamic_flops
    # For simplicity, we assume similar count for additions and ignore divisions (or add them as output elements).
    # Additions for depthwise and pointwise:
    dw_adds = ((kernel_size * kernel_size * in_channels) - 1) * out_height * out_width
    pw_adds = ((in_channels * out_channels) - 1) * out_height * out_width
    conv_adds = dw_adds + pw_adds
    # Divisions: assume one per output element for normalization:
    divs = out_height * out_width * out_channels

    total_flops = total_mults + conv_adds + divs

    output_shape = (out_height, out_width, out_channels)
    return total_params, total_flops, dw_mults, pw_mults, dynamic_flops, conv_adds, divs, output_shape

def benchmark_convolution(input_shape, out_channels, kernel_size=3, stride=2, num_kernels=4):
    """
    EXACT replication of the original benchmark_convolution function from DSKW.py
    """
    in_channels = input_shape[2]
    model = KWDSConv2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, num_kernels=num_kernels)
    model.eval()

    input_data = torch.randn(1, in_channels, input_shape[0], input_shape[1])
    
    # Warm-up
    with torch.no_grad():
        model(input_data)
    
    latencies = []
    for _ in range(100):
        start_time = time.time()
        with torch.no_grad():
            model(input_data)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)
    
    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency  = np.std(latencies)
    min_latency  = np.min(latencies)
    max_latency  = np.max(latencies)
    p95_latency  = np.percentile(latencies, 95)
    p99_latency  = np.percentile(latencies, 99)
    
    total_params, total_flops, dw_mults, pw_mults, dynamic_flops, conv_adds, divs, output_shape = calculate_flops_and_params(
        input_shape, out_channels, kernel_size, stride, num_kernels
    )
    
    print("\nBenchmarking Results for Depthwise Separable KWConv (KWDSConv2D):")
    print(f"  Output Channels: {out_channels}")
    print(f"  Kernel Size: {kernel_size}")
    print(f"  Stride: {stride}")
    print(f"  Output Shape: {output_shape}")
    print(f"  Parameters: {total_params / 1e3:.2f}K")
    print(f"  Total FLOPs: {total_flops / 1e6:.2f}M")
    print(f"    - Depthwise Mults: {dw_mults / 1e6:.2f}M")
    print(f"    - Pointwise Mults: {pw_mults / 1e6:.2f}M")
    print(f"    - Dynamic Branch FLOPs: {dynamic_flops / 1e6:.2f}M")
    print(f"    - Convolution Adds: {conv_adds / 1e6:.2f}M")
    print(f"    - Divisions: {divs / 1e6:.2f}M")
    print("Latency Statistics:")
    print(f"  Mean: {mean_latency:.2f}ms")
    print(f"  Std Dev: {std_latency:.2f}ms")
    print(f"  Min: {min_latency:.2f}ms | Max: {max_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms | P99: {p99_latency:.2f}ms")
    print("\n")

def run_original_dskw_benchmark():
    """
    EXACT replication of the original main execution from DSKW.py
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
    run_original_dskw_benchmark()
