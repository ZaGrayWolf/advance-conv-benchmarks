import math
import time
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.utils import _pair
from torchvision.ops import DeformConv2d

class DeformableConv2D(nn.Module):
    """
    Deformable Convolution v1 implementation.
    Uses torchvision's DeformConv2d with proper offset handling and comprehensive analysis.
    
    This implementation includes all the sophisticated analysis capabilities
    from the original DCNv1.py implementation.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1):
        super(DeformableConv2D, self).__init__()
        if padding is None:
            padding = kernel_size // 2
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.deform_conv = DeformConv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups
        )
        
    def calculate_parameters_and_flops(self, input_size):
        """
        Calculate parameters and FLOPs for deformable convolution.
        Exactly replicating the original DCNv1.py calculation methodology.
        
        Args:
            input_size: Input size (H, W) or single dimension H (assumes square)
            
        Returns:
            Dict containing comprehensive analysis
        """
        if isinstance(input_size, (list, tuple)):
            H_in, W_in = input_size[:2]
        else:
            H_in = W_in = input_size
            
        weight_params = self.out_channels * (self.in_channels // self.groups) * (self.kernel_size ** 2)
        bias_params = self.out_channels
        total_params = weight_params + bias_params

        H_out = math.floor((H_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
        W_out = math.floor((W_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)

        flops_per_output = 2 * ((self.in_channels // self.groups) * (self.kernel_size ** 2))
        total_output_elements = H_out * W_out * self.out_channels
        total_flops = flops_per_output * total_output_elements

        return {
            'total_params': total_params,
            'weight_params': weight_params,
            'bias_params': bias_params,
            'total_flops': total_flops,
            'flops_per_output': flops_per_output,
            'output_shape': (H_out, W_out, self.out_channels),
            'input_shape': (H_in, W_in, self.in_channels)
        }
        
    def forward(self, x, offset=None):
        if offset is None:
            # Generate zero offset if not provided
            B, C, H, W = x.shape
            H_out = math.floor((H + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
            W_out = math.floor((W + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
            
            # Offset channels: 2 coordinates (x,y) for each kernel position
            offset_channels = 2 * (self.kernel_size ** 2) * (self.in_channels // self.groups)
            offset = torch.zeros(B, offset_channels, H_out, W_out, device=x.device, dtype=x.dtype)
        
        return self.deform_conv(x, offset)
    
    def benchmark(self, input_tensor, num_runs=100, warmup_runs=5, detailed=True):
        """
        Comprehensive benchmarking with detailed latency statistics.
        Exactly replicating the original DCNv1.py benchmarking methodology.
        
        Args:
            input_tensor: Input tensor for benchmarking
            num_runs: Number of iterations for latency measurement
            warmup_runs: Number of warmup iterations
            detailed: Whether to return detailed analysis
            
        Returns:
            Dict containing comprehensive benchmark results
        """
        self.eval()
        device = next(self.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Calculate required offset tensor
        B, C, H, W = input_tensor.shape
        analysis = self.calculate_parameters_and_flops((H, W))
        H_out, W_out, _ = analysis['output_shape']
        
        offset_channels = 2 * (self.kernel_size ** 2) * (self.in_channels // self.groups)
        offset_tensor = torch.zeros(B, offset_channels, H_out, W_out, device=device, dtype=input_tensor.dtype)
        
        # Warm-up runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self(input_tensor, offset_tensor)
        
        # Benchmark latency
        latencies = []
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                _ = self(input_tensor, offset_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        latencies = np.array(latencies)
        
        # Calculate statistics exactly like original
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        results = {
            'mean_latency': mean_latency,
            'std_latency': std_latency,
            'min_latency': min_latency,
            'max_latency': max_latency,
            'p95_latency': p95_latency,
            'p99_latency': p99_latency,
            'all_latencies': latencies,
            'num_runs': num_runs,
            'input_shape': input_tensor.shape,
            'output_shape': self(input_tensor, offset_tensor).shape
        }
        
        if detailed:
            # Add FLOP analysis
            flop_analysis = self.calculate_parameters_and_flops((H, W))
            results.update(flop_analysis)
        
        return results
    
    def print_detailed_analysis(self, input_shape, style="original"):
        """
        Print detailed analysis in the original DCNv1.py style.
        
        Args:
            input_shape: Input tensor shape for analysis
            style: Output style ("original", "summary", or "comprehensive")
        """
        if len(input_shape) == 4:
            # (B, C, H, W) format
            H, W = input_shape[2], input_shape[3]
        elif len(input_shape) == 3:
            # (H, W, C) or (C, H, W) format
            if input_shape[2] == self.in_channels:
                H, W = input_shape[0], input_shape[1]
            else:
                H, W = input_shape[1], input_shape[2]
        else:
            H, W = input_shape[0], input_shape[1]
            
        analysis = self.calculate_parameters_and_flops((H, W))
        
        if style == "original":
            print("Deformable Convolution Layer:")
            print(f"Input Shape: (1, {self.in_channels}, {H}, {W})")
            print(f"Output Shape: (1, {self.out_channels}, {analysis['output_shape'][0]}, {analysis['output_shape'][1]})")
            print(f"Total Parameters: {analysis['total_params']} ({analysis['total_params'] / 1e3:.2f}K)")
            print(f"Estimated FLOPs: {analysis['total_flops']} (~{analysis['total_flops'] / 1e6:.2f}M)")
        
        elif style == "summary":
            print(f"DeformableConv2D: {analysis['total_params']/1e3:.1f}K params, {analysis['total_flops']/1e6:.1f}M FLOPs")
        
        elif style == "comprehensive":
            self.print_detailed_analysis(input_shape, "original")
            print(f"Parameter Breakdown:")
            print(f"  - Weight Parameters: {analysis['weight_params']}")
            print(f"  - Bias Parameters: {analysis['bias_params']}")
            print(f"FLOPs Breakdown:")
            print(f"  - FLOPs per Output Element: {analysis['flops_per_output']}")
            print(f"  - Total Output Elements: {analysis['output_shape'][0] * analysis['output_shape'][1] * self.out_channels}")
    
    def benchmark_and_print(self, input_shape, num_runs=100, output_channels=None):
        """
        Run benchmark and print results in original DCNv1.py style.
        
        Args:
            input_shape: Input tensor shape (H, W, C) or (B, C, H, W)
            num_runs: Number of benchmark iterations
            output_channels: For display purposes
        """
        if len(input_shape) == 3:
            # (H, W, C) format
            input_tensor = torch.randn(1, input_shape[2], input_shape[0], input_shape[1])
        elif len(input_shape) == 4:
            input_tensor = torch.randn(*input_shape)
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")
        
        results = self.benchmark(input_tensor, num_runs=num_runs, detailed=True)
        
        print("Deformable Convolution Layer:")
        print(f"Input Shape: {results['input_shape']}")
        print(f"Output Shape: {results['output_shape']}")
        print(f"Total Parameters: {results['total_params']} ({results['total_params'] / 1e3:.2f}K)")
        print(f"Estimated FLOPs: {results['total_flops']} (~{results['total_flops'] / 1e6:.2f}M)")
        print("Latency Statistics:")
        print(f"  Mean: {results['mean_latency']:.2f}ms")
        print(f"  Std Dev: {results['std_latency']:.2f}ms")
        print(f"  Min: {results['min_latency']:.2f}ms")
        print(f"  Max: {results['max_latency']:.2f}ms")
        print(f"  P95: {results['p95_latency']:.2f}ms")
        print(f"  P99: {results['p99_latency']:.2f}ms")
        print()
        
        return results


def calculate_parameters_and_flops_standalone(in_channels, out_channels, kernel_size, input_size, stride, padding, dilation, groups):
    """
    Standalone function that exactly replicates the original DCNv1.py calculate_parameters_and_flops function.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size
        input_size: Input spatial size
        stride: Stride
        padding: Padding
        dilation: Dilation
        groups: Groups
        
    Returns:
        Tuple of (total_params, total_flops, output_shape)
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


def benchmark_deformable_convolution_original_style(input_shape=(224, 224, 3), output_channels_list=[64, 128, 256], **kwargs):
    """
    Replicate the original DCNv1.py benchmarking exactly.
    
    Args:
        input_shape: Input shape (H, W, C)
        output_channels_list: List of output channel configurations to test
        **kwargs: Additional arguments for DeformableConv2D
    """
    print("=" * 60)
    print("DEFORMABLE CONVOLUTION v1 ANALYSIS")
    print("=" * 60)
    
    for out_channels in output_channels_list:
        print(f"\nBenchmarking with output channels = {out_channels} ({'Small' if out_channels == 64 else 'Medium' if out_channels == 128 else 'Large'} Output):")
        
        model = DeformableConv2D(
            in_channels=input_shape[2], 
            out_channels=out_channels,
            kernel_size=kwargs.get('kernel_size', 3),
            stride=kwargs.get('stride', 1),
            padding=kwargs.get('padding', 1),
            dilation=kwargs.get('dilation', 1),
            groups=kwargs.get('groups', 1)
        )
        
        model.benchmark_and_print(input_shape, output_channels=out_channels)
