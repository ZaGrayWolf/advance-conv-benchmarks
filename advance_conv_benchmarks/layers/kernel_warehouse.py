import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

class KWDSConv2D(nn.Module):
    """
    Dynamic Depthwise Separable Convolution (Kernel Warehouse version).
    Generates a dynamic depthwise kernel for each input channel via a kernel bank
    and an attention mechanism, then applies a fixed pointwise convolution.
    
    This implementation includes all the sophisticated analysis capabilities
    from the original DSKW.py implementation.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, num_kernels=4):
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
    
    def calculate_flops_and_params(self, input_shape):
        """
        Calculate FLOPs and parameters for Kernel Warehouse Depthwise Separable convolution.
        Exactly replicating the original DSKW.py calculation with full breakdown.
        
        Args:
            input_shape: Input tensor shape (H, W, C) or (B, C, H, W)
            
        Returns:
            Dict containing comprehensive analysis with detailed breakdown
        """
        if len(input_shape) == 4:
            # (B, C, H, W) format
            _, in_channels, height, width = input_shape
            input_shape_hwc = (height, width, in_channels)
        elif len(input_shape) == 3 and input_shape[2] == self.in_channels:
            # (H, W, C) format
            input_shape_hwc = input_shape
        elif len(input_shape) == 3 and input_shape[0] == self.in_channels:
            # (C, H, W) format
            input_shape_hwc = (input_shape[1], input_shape[2], input_shape[0])
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")
        
        stride = self.stride[0] if isinstance(self.stride, tuple) else self.stride
        
        # Output spatial dimensions (assuming padding = kernel_size//2)
        out_height = (input_shape_hwc[0] + 2*(self.kernel_size//2) - self.kernel_size) // stride + 1
        out_width = (input_shape_hwc[1] + 2*(self.kernel_size//2) - self.kernel_size) // stride + 1

        # -------------------------
        # Depthwise Branch
        # -------------------------
        # Depthwise kernel bank parameters:
        # Each candidate has: (kernel_size^2 * 1 * in_channels) parameters (ignoring bias for simplicity)
        # Total depthwise params = num_kernels * in_channels * kernel_size^2.
        dw_params = self.num_kernels * self.in_channels * (self.kernel_size * self.kernel_size)
        
        # Pointwise conv parameters:
        # (in_channels * out_channels) + out_channels (bias)
        pw_params = (self.in_channels * self.out_channels) + self.out_channels
        
        # Dynamic branch parameters (attention):
        # FC: in_channels -> num_kernels: (in_channels * num_kernels) + num_kernels
        attn_params = (self.in_channels * self.num_kernels) + self.num_kernels

        total_params = dw_params + pw_params + attn_params

        # -------------------------
        # FLOPs Estimation:
        # Depthwise conv FLOPs:
        # For each output element, depthwise multiplications: (kernel_size^2 * in_channels) operations over all channels.
        dw_mults = (self.kernel_size * self.kernel_size * self.in_channels) * out_height * out_width
        # Pointwise conv FLOPs:
        pw_mults = (self.in_channels * self.out_channels) * out_height * out_width
        # Dynamic branch FLOPs:
        # GAP: in_channels * (H*W) (roughly)
        gap_flops = self.in_channels * (input_shape_hwc[0] * input_shape_hwc[1])
        # FC layer: mapping in_channels -> num_kernels ~ 2 * in_channels * num_kernels
        fc_flops = 2 * self.in_channels * self.num_kernels
        # Softmax: assume ~2 * num_kernels
        softmax_flops = 2 * self.num_kernels
        # Kernel weighting: For each element in the depthwise kernel bank
        kernel_elements = self.in_channels * (self.kernel_size * self.kernel_size)
        weighting_flops = (self.num_kernels + (self.num_kernels - 1)) * kernel_elements

        dynamic_flops = gap_flops + fc_flops + softmax_flops + weighting_flops

        # Total multiplications:
        total_mults = dw_mults + pw_mults + dynamic_flops
        # For simplicity, we assume similar count for additions and ignore divisions (or add them as output elements).
        # Additions for depthwise and pointwise:
        dw_adds = ((self.kernel_size * self.kernel_size * self.in_channels) - 1) * out_height * out_width
        pw_adds = ((self.in_channels * self.out_channels) - 1) * out_height * out_width
        conv_adds = dw_adds + pw_adds
        # Divisions: assume one per output element for normalization:
        divs = out_height * out_width * self.out_channels

        total_flops = total_mults + conv_adds + divs

        output_shape = (out_height, out_width, self.out_channels)
        
        return {
            'total_params': total_params,
            'total_flops': total_flops,
            'dw_params': dw_params,
            'pw_params': pw_params,
            'attn_params': attn_params,
            'dw_mults': dw_mults,
            'pw_mults': pw_mults,
            'dynamic_flops': dynamic_flops,
            'conv_adds': conv_adds,
            'divisions': divs,
            'output_shape': output_shape,
            'gap_flops': gap_flops,
            'fc_flops': fc_flops,
            'softmax_flops': softmax_flops,
            'weighting_flops': weighting_flops
        }
    
    def benchmark(self, input_tensor, num_runs=100, warmup_runs=5, detailed=True):
        """
        Comprehensive benchmarking with detailed latency statistics.
        Exactly replicating the original DSKW.py benchmarking methodology.
        
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
        
        # Warm-up runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self(input_tensor)
        
        # Benchmark latency
        latencies = []
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                _ = self(input_tensor)
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
            'num_runs': num_runs
        }
        
        if detailed:
            # Add FLOP analysis
            flop_analysis = self.calculate_flops_and_params(input_tensor.shape)
            results.update(flop_analysis)
        
        return results
    
    def print_detailed_analysis(self, input_shape, style="original"):
        """
        Print detailed analysis in the original DSKW.py style.
        
        Args:
            input_shape: Input tensor shape for analysis
            style: Output style ("original", "summary", or "comprehensive")
        """
        analysis = self.calculate_flops_and_params(input_shape)
        
        if style == "original":
            print("\nDetailed Analysis for Depthwise Separable KWConv (KWDSConv2D):")
            print(f"  Kernel Size: {self.kernel_size}")
            print(f"  Stride: {self.stride}")
            print(f"  Num Kernels: {self.num_kernels}")
            print(f"  Output Shape: {analysis['output_shape']}")
            print(f"  Parameters: {analysis['total_params'] / 1e3:.2f}K")
            print(f"  Total FLOPs: {analysis['total_flops'] / 1e6:.2f}M")
            print(f"    - Depthwise Mults: {analysis['dw_mults'] / 1e6:.2f}M")
            print(f"    - Pointwise Mults: {analysis['pw_mults'] / 1e6:.2f}M")
            print(f"    - Dynamic Branch FLOPs: {analysis['dynamic_flops'] / 1e6:.2f}M")
            print(f"    - Convolution Adds: {analysis['conv_adds'] / 1e6:.2f}M")
            print(f"    - Divisions: {analysis['divisions'] / 1e6:.2f}M")
            print("\nParameter Breakdown:")
            print(f"  - Depthwise Kernel Bank: {analysis['dw_params']}")
            print(f"  - Pointwise Conv: {analysis['pw_params']}")
            print(f"  - Attention Branch: {analysis['attn_params']}")
        
        elif style == "summary":
            print(f"KWDSConv2D: {analysis['total_params']/1e3:.1f}K params, {analysis['total_flops']/1e6:.1f}M FLOPs")
        
        elif style == "comprehensive":
            self.print_detailed_analysis(input_shape, "original")
            print("\nDynamic Branch Breakdown:")
            print(f"  - GAP FLOPs: {analysis['gap_flops'] / 1e6:.2f}M")
            print(f"  - FC FLOPs: {analysis['fc_flops'] / 1e6:.2f}M")
            print(f"  - Softmax FLOPs: {analysis['softmax_flops'] / 1e6:.2f}M")
            print(f"  - Kernel Weighting FLOPs: {analysis['weighting_flops'] / 1e6:.2f}M")
    
    def benchmark_and_print(self, input_shape, num_runs=100, output_channels=None):
        """
        Run benchmark and print results in original DSKW.py style.
        
        Args:
            input_shape: Input tensor shape (H, W, C)
            num_runs: Number of benchmark iterations
            output_channels: For display purposes
        """
        if len(input_shape) == 3:
            # (H, W, C) format
            input_tensor = torch.randn(1, input_shape[2], input_shape[0], input_shape[1])
        else:
            input_tensor = torch.randn(*input_shape) if len(input_shape) == 4 else torch.randn(1, *input_shape)
        
        results = self.benchmark(input_tensor, num_runs=num_runs, detailed=True)
        
        print("\nBenchmarking Results for Depthwise Separable KWConv (KWDSConv2D):")
        if output_channels:
            print(f"  Output Channels: {output_channels}")
        print(f"  Kernel Size: {self.kernel_size}")
        print(f"  Stride: {self.stride}")
        print(f"  Output Shape: {results['output_shape']}")
        print(f"  Parameters: {results['total_params'] / 1e3:.2f}K")
        print(f"  Total FLOPs: {results['total_flops'] / 1e6:.2f}M")
        print(f"    - Depthwise Mults: {results['dw_mults'] / 1e6:.2f}M")
        print(f"    - Pointwise Mults: {results['pw_mults'] / 1e6:.2f}M")
        print(f"    - Dynamic Branch FLOPs: {results['dynamic_flops'] / 1e6:.2f}M")
        print(f"    - Convolution Adds: {results['conv_adds'] / 1e6:.2f}M")
        print(f"    - Divisions: {results['divisions'] / 1e6:.2f}M")
        print("Latency Statistics:")
        print(f"  Mean: {results['mean_latency']:.2f}ms")
        print(f"  Std Dev: {results['std_latency']:.2f}ms")
        print(f"  Min: {results['min_latency']:.2f}ms | Max: {results['max_latency']:.2f}ms")
        print(f"  P95: {results['p95_latency']:.2f}ms | P99: {results['p99_latency']:.2f}ms")
        print()
        
        return results


def benchmark_kwds_conv_original_style(input_shape=(224, 224, 3), output_channels_list=[64, 128, 256], **kwargs):
    """
    Replicate the original DSKW.py benchmarking exactly.
    
    Args:
        input_shape: Input shape (H, W, C)
        output_channels_list: List of output channel configurations to test
        **kwargs: Additional arguments for KWDSConv2D
    """
    print("=" * 60)
    print("KERNEL WAREHOUSE DEPTHWISE SEPARABLE CONVOLUTION ANALYSIS")
    print("=" * 60)
    
    for out_channels in output_channels_list:
        print(f"\nBenchmarking with output channels = {out_channels}:")
        
        model = KWDSConv2D(
            in_channels=input_shape[2], 
            out_channels=out_channels,
            kernel_size=kwargs.get('kernel_size', 3),
            stride=kwargs.get('stride', 2),
            num_kernels=kwargs.get('num_kernels', 4)
        )
        
        model.benchmark_and_print(input_shape, output_channels=out_channels)
