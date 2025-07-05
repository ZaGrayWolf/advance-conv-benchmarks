"""
EXACT replication of DSODConv.py with all the depth and comprehensiveness.
This module preserves every detail of your original ODConv implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

class ODConv2D(nn.Module):
    """
    Omni-Dimensional Dynamic Convolution (ODConv) with Depthwise Separable Convolution.
    
    This layer generates a dynamic depthwise kernel by applying dynamic attention
    across four dimensions:
      - Spatial (αₛ): shape [B, 1, 1, K, K]
      - Input Channel (α𝑐): shape [B, 1, C_in, 1, 1]
      - Kernel Weight (α𝑤): shape [B, C_in, 1, K, K]
      - Candidate Kernel Weights (αᴄ): weights over 6 candidate depthwise kernels, shape [B, 6, 1, 1, 1]
    
    The base candidate kernels are stored in a bank of shape 
    [6, C_in, 1, K, K]. They are aggregated (via a weighted sum using αᴄ) and then 
    modulated elementwise by the other attentions to form the final dynamic depthwise kernel.
    
    This dynamic kernel is applied in a depthwise convolution (groups = C_in), 
    and its output is then passed through a fixed pointwise convolution.
    
    An additional output channel attention is applied to the pointwise output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, reduction_ratio=4, num_candidates=6):
        super(ODConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # assume square kernel
        self.stride = stride
        self.num_candidates = num_candidates
        
        # Candidate depthwise kernels: shape [num_candidates, in_channels, 1, kernel_size, kernel_size]
        self.dw_kernel_bank = nn.Parameter(torch.randn(num_candidates, in_channels, 1, kernel_size, kernel_size))
        
        # Fixed pointwise convolution: maps from in_channels to out_channels.
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Global Average Pooling to extract a context vector [B, in_channels]
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Attention branches:
        # 1. Spatial attention: generate K*K values.
        self.fc_spatial = nn.Linear(in_channels, kernel_size * kernel_size)
        # 2. Input channel attention: generate in_channels values.
        self.fc_in = nn.Linear(in_channels, in_channels)
        # 3. Kernel weight attention: generate (C_in * K*K) values.
        self.fc_kernel = nn.Linear(in_channels, in_channels * kernel_size * kernel_size)
        # 4. Candidate kernel weights: generate num_candidates values.
        self.fc_candidate = nn.Linear(in_channels, num_candidates)
        # 5. Output channel attention (for pointwise branch): generate out_channels values.
        self.fc_out = nn.Linear(in_channels, out_channels)
        
        # Activations
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)  # for candidate weights

    def forward(self, x):
        batch, channels, H, W = x.size()
        
        # Get global context: [B, C_in]
        context = self.global_avg_pool(x).view(batch, self.in_channels)
        
        # Generate dynamic attention factors:
        # Spatial attention: [B, K*K] -> reshape to [B, 1, 1, K, K]
        attn_spatial = self.sigmoid(self.fc_spatial(context)).view(batch, 1, 1, self.kernel_size, self.kernel_size)
        # Input channel attention: [B, C_in] -> reshape to [B, 1, C_in, 1, 1]
        attn_in = self.sigmoid(self.fc_in(context)).view(batch, 1, self.in_channels, 1, 1)
        # Kernel weight attention: [B, (C_in*K*K)] -> reshape to [B, C_in, 1, K, K]
        attn_kernel = self.sigmoid(self.fc_kernel(context)).view(batch, self.in_channels, 1, self.kernel_size, self.kernel_size)
        # Candidate kernel weights: [B, num_candidates] -> softmax -> reshape to [B, num_candidates, 1, 1, 1]
        attn_candidate = self.softmax(self.fc_candidate(context)).view(batch, self.num_candidates, 1, 1, 1, 1)
        # Output channel attention for pointwise conv: [B, out_channels] -> reshape to [B, out_channels, 1, 1]
        attn_out = self.sigmoid(self.fc_out(context)).view(batch, self.out_channels, 1, 1)
        
        # Aggregate candidate depthwise kernels:
        # dw_kernel_bank: [num_candidates, in_channels, 1, K, K] -> unsqueeze to [1, num_candidates, in_channels, 1, K, K]
        # Multiply by candidate weights and sum over the candidate dimension -> [B, in_channels, 1, K, K]
        dynamic_dw_kernel = (self.dw_kernel_bank.unsqueeze(0) * attn_candidate).sum(dim=1)
        
        # Now apply additional dynamic factors to the aggregated kernel:
        # dynamic_dw_kernel: [B, in_channels, 1, K, K]
        # Multiply elementwise by spatial, input channel, and kernel weight attentions.
        # attn_spatial: [B, 1, 1, K, K] -> broadcast to [B, in_channels, 1, K, K]
        # attn_in: [B, 1, C_in, 1, 1] -> need to match channel dim: it can be transposed to [B, C_in, 1, 1, 1] then broadcast.
        attn_in = attn_in.transpose(1,2)  # now shape [B, C_in, 1, 1, 1]
        # attn_kernel: [B, C_in, 1, K, K]
        dynamic_dw_kernel = dynamic_dw_kernel * attn_spatial * attn_in * attn_kernel  # [B, C_in, 1, K, K]
        
        # For batch size 1, remove batch dimension to match F.conv2d requirements.
        if batch == 1:
            dynamic_dw_kernel = dynamic_dw_kernel.squeeze(0)  # now shape [C_in, 1, K, K]
        
        # Apply dynamic depthwise convolution with groups = in_channels.
        out_dw = F.conv2d(x, dynamic_dw_kernel, stride=self.stride, padding=self.kernel_size // 2, groups=self.in_channels)
        
        # Apply fixed pointwise convolution.
        out_pw = self.pointwise(out_dw)
        # Apply output channel attention.
        out = out_pw * attn_out
        return out

def calculate_flops_and_params(input_shape, out_channels, kernel_size, stride, in_channels, num_candidates=6):
    """
    EXACT replication of the original calculate_flops_and_params function from DSODConv.py
    Rough estimation of parameters and FLOPs:
      - Depthwise candidate kernel bank: num_candidates * (in_channels * kernel_size^2)
      - Attention branch: FC layers for spatial, input, kernel, candidate, and output attentions.
      - Pointwise conv: (in_channels * out_channels) + out_channels
    """
    # Candidate kernel bank parameters:
    dw_bank_params = num_candidates * in_channels * (kernel_size * kernel_size)
    # Pointwise conv parameters:
    pw_params = in_channels * out_channels + out_channels
    # Attention branch parameters (approximate):
    fc_spatial_params = in_channels * (kernel_size * kernel_size)
    fc_in_params = in_channels * in_channels
    fc_kernel_params = in_channels * (in_channels * kernel_size * kernel_size)
    fc_candidate_params = in_channels * num_candidates + num_candidates
    fc_out_params = in_channels * out_channels
    attn_params = fc_spatial_params + fc_in_params + fc_kernel_params + fc_candidate_params + fc_out_params
    total_params = dw_bank_params + pw_params + attn_params

    # Compute output spatial dimensions (assuming padding = kernel_size//2)
    H_out = (input_shape[0] + 2*(kernel_size//2) - kernel_size) // stride + 1
    W_out = (input_shape[1] + 2*(kernel_size//2) - kernel_size) // stride + 1

    # FLOPs (dummy estimation):
    # Depthwise conv FLOPs:
    dw_mults = (kernel_size * kernel_size * in_channels) * H_out * W_out
    # Pointwise conv FLOPs:
    pw_mults = (in_channels * out_channels) * H_out * W_out
    # Convolution additions:
    dw_adds = ((kernel_size * kernel_size * in_channels) - 1) * H_out * W_out
    pw_adds = ((in_channels * out_channels) - 1) * H_out * W_out
    conv_adds = dw_adds + pw_adds
    # Divisions:
    divs = H_out * W_out * out_channels

    # Dynamic branch FLOPs (rough estimation):
    gap_flops = in_channels * (input_shape[0] * input_shape[1])
    fc_flops = (in_channels * (kernel_size*kernel_size) + in_channels * in_channels + in_channels * num_candidates + in_channels * out_channels)
    softmax_flops = 2 * num_candidates
    kernel_elements = in_channels * (kernel_size * kernel_size)
    weighting_flops = (num_candidates + (num_candidates - 1)) * kernel_elements
    dynamic_flops = gap_flops + fc_flops + softmax_flops + weighting_flops

    total_mults = dw_mults + pw_mults + dynamic_flops
    total_flops = total_mults + conv_adds + divs
    output_shape = (H_out, W_out, out_channels)
    return total_params, total_flops, dw_mults, pw_mults, dynamic_flops, conv_adds, divs, output_shape

def benchmark_convolution(input_shape, out_channels, kernel_size=3, stride=2, num_candidates=6):
    """
    EXACT replication of the original benchmark_convolution function from DSODConv.py
    """
    in_channels = input_shape[2]
    model = ODConv2D(in_channels, out_channels, kernel_size, stride, reduction_ratio=4, num_candidates=num_candidates)
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
        input_shape, out_channels, kernel_size, stride, in_channels, num_candidates
    )
    
    print("\nBenchmarking Results for ODConv with Depthwise Separable Decomposition and Dynamic Attention:")
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

def run_original_dsodconv_benchmark():
    """
    EXACT replication of the original main execution from DSODConv.py
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
    run_original_dsodconv_benchmark()
