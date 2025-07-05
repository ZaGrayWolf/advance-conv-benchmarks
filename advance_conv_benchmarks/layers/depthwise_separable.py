import torch
import torch.nn as nn
import time
import numpy as np

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution with comprehensive analysis capabilities.
    
    This implementation includes all the sophisticated analysis capabilities
    from the original DS_Ptflops.py implementation.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None):
        super(DepthwiseSeparableConv, self).__init__()
        if padding is None:
            padding = kernel_size // 2
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
    def calculate_flops_and_params(self, input_shape):
        """
        Calculate FLOPs and parameters for depthwise separable convolution.
        Exactly replicating the original DS_Ptflops.py calculation methodology.
        
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
        
        # Output spatial dimensions
        if isinstance(self.stride, (list, tuple)):
            stride_h, stride_w = self.stride[0], self.stride[1]
        else:
            stride_h = stride_w = self.stride
            
        if isinstance(self.kernel_size, (list, tuple)):
            kernel_h, kernel_w = self.kernel_size[0], self.kernel_size[1]
        else:
            kernel_h = kernel_w = self.kernel_size
            
        out_height = (input_shape_hwc[0] + 2*self.padding - kernel_h) // stride_h + 1
        out_width = (input_shape_hwc[1] + 2*self.padding - kernel_w) // stride_w + 1
        
        # Depthwise convolution parameters: in_channels * kernel_size^2 (no bias)
        dw_params = self.in_channels * (kernel_h * kernel_w)
        
        # Pointwise convolution parameters: in_channels * out_channels (no bias)
        pw_params = self.in_channels * self.out_channels
        
        total_params = dw_params + pw_params
        
        # Depthwise convolution FLOPs
        # Each output element requires kernel_size^2 operations for one channel
        # Total: kernel_size^2 * output_height * output_width * in_channels
        dw_flops = (kernel_h * kernel_w) * out_height * out_width * self.in_channels
        
        # Pointwise convolution FLOPs
        # Each output element requires in_channels operations
        # Total: in_channels * output_height * output_width * out_channels
        pw_flops = self.in_channels * self.out_channels * out_height * out_width
        
        total_flops = dw_flops + pw_flops
        
        # MACs calculation (for compatibility with ptflops)
        dw_macs = (kernel_h * kernel_w) * out_height * out_width * self.in_channels
        pw_macs = self.in_channels * self.out_channels * out_height * out_width
        total_macs = dw_macs + pw_macs
        
        # Alternative FLOP calculation (MACs * 2)
        flops_from_macs = total_macs * 2
        
        return {
            'total_params': total_params,
            'dw_params': dw_params,
            'pw_params': pw_params,
            'total_flops': total_flops,
            'dw_flops': dw_flops,
            'pw_flops': pw_flops,
            'total_macs': total_macs,
            'dw_macs': dw_macs,
            'pw_macs': pw_macs,
            'flops_from_macs': flops_from_macs,
            'output_shape': (out_height, out_width, self.out_channels),
            'depthwise_output_shape': (out_height, out_width, self.in_channels),
            'pointwise_output_shape': (out_height, out_width, self.out_channels)
        }
    
    def benchmark(self, input_tensor, num_runs=100, warmup_runs=5, detailed=True):
        """
        Comprehensive benchmarking with detailed latency statistics.
        Exactly replicating the original DS_Ptflops.py benchmarking methodology.
        
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
        
        # Warm-up runs for any lazy initialization
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
        Print detailed analysis in the original DS_Ptflops.py style.
        
        Args:
            input_shape: Input tensor shape for analysis
            style: Output style ("original", "summary", or "comprehensive")
        """
        analysis = self.calculate_flops_and_params(input_shape)
        
        if style == "original":
            print(f"Depthwise Separable Convolution:")
            print(f"Kernel Size: {self.kernel_size}")
            print(f"Input Shape: {input_shape}")
            print(f"Output Shape: {analysis['output_shape']}")
            print(f"Parameters: {analysis['total_params'] / 1e3:.2f}K")
            print(f"Total FLOPs: {analysis['total_flops'] / 1e6:.2f}M")
            print(f"  - Depthwise FLOPs: {analysis['dw_flops'] / 1e6:.2f}M")
            print(f"  - Pointwise FLOPs: {analysis['pw_flops'] / 1e6:.2f}M")
        
        elif style == "summary":
            print(f"DepthwiseSeparableConv: {analysis['total_params']/1e3:.1f}K params, {analysis['total_flops']/1e6:.1f}M FLOPs")
        
        elif style == "comprehensive":
            self.print_detailed_analysis(input_shape, "original")
            print(f"Parameter Breakdown:")
            print(f"  - Depthwise Parameters: {analysis['dw_params']}")
            print(f"  - Pointwise Parameters: {analysis['pw_params']}")
            print(f"MACs Breakdown:")
            print(f"  - Depthwise MACs: {analysis['dw_macs'] / 1e6:.2f}M")
            print(f"  - Pointwise MACs: {analysis['pw_macs'] / 1e6:.2f}M")
            print(f"  - Total MACs: {analysis['total_macs'] / 1e6:.2f}M")
            print(f"Output Shape Details:")
            print(f"  - After Depthwise: {analysis['depthwise_output_shape']}")
            print(f"  - After Pointwise: {analysis['pointwise_output_shape']}")
    
    def benchmark_and_print(self, input_shape, num_runs=100, output_channels=None):
        """
        Run benchmark and print results in original DS_Ptflops.py style.
        
        Args:
            input_shape: Input tensor shape
            num_runs: Number of benchmark iterations
            output_channels: For display purposes
        """
        if len(input_shape) == 3:
            # (H, W, C) format
            input_tensor = torch.randn(1, input_shape[2], input_shape[0], input_shape[1])
        else:
            input_tensor = torch.randn(*input_shape) if len(input_shape) == 4 else torch.randn(1, *input_shape)
        
        results = self.benchmark(input_tensor, num_runs=num_runs, detailed=True)
        
        print(f"Depthwise Separable Convolution:")
        print(f"Kernel Size: {self.kernel_size}")
        print(f"Input Shape: {input_shape}")
        print(f"Output Shape: {results['output_shape']}")
        print(f"Parameters: {results['total_params'] / 1e3:.2f}K")
        print(f"Total FLOPs: {results['total_flops'] / 1e6:.2f}M")
        print("Latency Statistics:")
        print(f"  Mean: {results['mean_latency']:.2f}ms")
        print(f"  Std Dev: {results['std_latency']:.2f}ms")
        print(f"  Min: {results['min_latency']:.2f}ms")
        print(f"  Max: {results['max_latency']:.2f}ms")
        print(f"  P95: {results['p95_latency']:.2f}ms")
        print(f"  P99: {results['p99_latency']:.2f}ms")
        print()
        
        return results


def benchmark_depthwise_separable_original_style(input_shape=(224, 224, 3), output_channels_list=[64, 128, 256], **kwargs):
    """
    Replicate the original DS_Ptflops.py benchmarking exactly.
    
    Args:
        input_shape: Input shape (H, W, C)
        output_channels_list: List of output channel configurations to test
        **kwargs: Additional arguments for DepthwiseSeparableConv
    """
    print("=" * 60)
    print("DEPTHWISE SEPARABLE CONVOLUTION ANALYSIS")
    print("=" * 60)
    
    for out_channels in output_channels_list:
        print(f"\nBenchmarking with output channels = {out_channels} ({'Small' if out_channels == 64 else 'Medium' if out_channels == 128 else 'Large'} Output):")
        
        model = DepthwiseSeparableConv(
            in_channels=input_shape[2], 
            out_channels=out_channels,
            kernel_size=kwargs.get('kernel_size', 3),
            stride=kwargs.get('stride', 2),
            padding=kwargs.get('padding', None)
        )
        
        model.benchmark_and_print(input_shape, output_channels=out_channels)
