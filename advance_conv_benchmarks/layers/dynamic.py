import torch
import torch.nn as nn
import time
import numpy as np

class DynamicConv2D(nn.Module):
    """
    Dynamic Convolution 2D with comprehensive analysis capabilities.
    
    This implementation includes all the sophisticated analysis capabilities
    from the original D_Ptflops.py implementation.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super(DynamicConv2D, self).__init__()
        if padding is None:
            padding = kernel_size // 2
            
        self.in_channels = in_channels
        self.out_channels = out_channels  
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.kernel_weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.kernel_weights, bias=self.bias, stride=self.stride, padding=self.padding)
    
    def calculate_flops_and_params(self, input_shape):
        """
        Calculate FLOPs and parameters for dynamic convolution.
        
        Args:
            input_shape: Input tensor shape (H, W, C) or (B, C, H, W)
            
        Returns:
            Dict containing comprehensive analysis
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
        
        # Parameters calculation
        weight_params = self.out_channels * self.in_channels * (self.kernel_size ** 2)
        bias_params = self.out_channels
        total_params = weight_params + bias_params
        
        # Output spatial dimensions
        out_height = (input_shape_hwc[0] + 2*self.padding - self.kernel_size) // self.stride + 1
        out_width = (input_shape_hwc[1] + 2*self.padding - self.kernel_size) // self.stride + 1
        
        # FLOPs calculation
        # Each output element requires: kernel_size^2 * in_channels multiplications
        # Plus kernel_size^2 * in_channels - 1 additions
        mults_per_output = self.kernel_size * self.kernel_size * self.in_channels
        adds_per_output = (self.kernel_size * self.kernel_size * self.in_channels) - 1
        total_output_elements = out_height * out_width * self.out_channels
        
        total_mults = mults_per_output * total_output_elements
        total_adds = adds_per_output * total_output_elements
        total_flops = total_mults + total_adds
        
        # MACs calculation
        macs = mults_per_output * total_output_elements
        flops_from_macs = macs * 2
        
        return {
            'total_params': total_params,
            'weight_params': weight_params,
            'bias_params': bias_params,
            'total_flops': total_flops,
            'total_mults': total_mults,
            'total_adds': total_adds,
            'macs': macs,
            'flops_from_macs': flops_from_macs,
            'output_shape': (out_height, out_width, self.out_channels),
            'mults_per_output': mults_per_output,
            'adds_per_output': adds_per_output,
            'total_output_elements': total_output_elements
        }
    
    def benchmark(self, input_tensor, num_runs=100, warmup_runs=5, detailed=True):
        """
        Comprehensive benchmarking with detailed latency statistics.
        
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
        
        # Calculate statistics
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
            flop_analysis = self.calculate_flops_and_params(input_tensor.shape)
            results.update(flop_analysis)
        
        return results
    
    def print_detailed_analysis(self, input_shape, style="original"):
        """
        Print detailed analysis in the original D_Ptflops.py style.
        
        Args:
            input_shape: Input tensor shape for analysis
            style: Output style ("original", "summary", or "comprehensive")
        """
        analysis = self.calculate_flops_and_params(input_shape)
        
        if style == "original":
            print(f"Dynamic Convolution")
            print(f"Kernel Size: {self.kernel_size}x{self.kernel_size}")
            print(f"Input Shape: {input_shape}")
            print(f"Output Channels: {self.out_channels}")
            print(f"Parameters: {analysis['total_params'] / 1e3:.2f}K")
            print(f"MACs: {analysis['macs'] / 1e6:.2f}M")
            print(f"FLOPs: {analysis['flops_from_macs'] / 1e6:.2f}M")
        
        elif style == "summary":
            print(f"DynamicConv2D: {analysis['total_params']/1e3:.1f}K params, {analysis['total_flops']/1e6:.1f}M FLOPs")
        
        elif style == "comprehensive":
            self.print_detailed_analysis(input_shape, "original")
            print(f"Parameter Breakdown:")
            print(f"  - Weight Parameters: {analysis['weight_params']}")
            print(f"  - Bias Parameters: {analysis['bias_params']}")
            print(f"FLOP Breakdown:")
            print(f"  - Total Multiplications: {analysis['total_mults'] / 1e6:.2f}M")
            print(f"  - Total Additions: {analysis['total_adds'] / 1e6:.2f}M")
            print(f"  - Multiplications per Output: {analysis['mults_per_output']}")
            print(f"  - Total Output Elements: {analysis['total_output_elements']}")
    
    def benchmark_and_print(self, input_shape, num_runs=100, output_channels=None):
        """
        Run benchmark and print results in original D_Ptflops.py style.
        
        Args:
            input_shape: Input tensor shape
            num_runs: Number of benchmark iterations
            output_channels: For display purposes
        """
        if len(input_shape) == 3:
            # Assume (C, H, W) format for dynamic conv
            input_tensor = torch.randn(1, *input_shape)
        else:
            input_tensor = torch.randn(*input_shape) if len(input_shape) == 4 else torch.randn(1, *input_shape)
        
        results = self.benchmark(input_tensor, num_runs=num_runs, detailed=True)
        
        print(f"Dynamic Convolution")
        print(f"Kernel Size: {self.kernel_size}x{self.kernel_size}")
        print(f"Input Shape: {input_shape}")
        if output_channels:
            print(f"Output Channels: {output_channels}")
        print(f"Parameters: {results['total_params'] / 1e3:.2f}K")
        print(f"MACs: {results['macs'] / 1e6:.2f}M")
        print("Latency Statistics:")
        print(f"  Mean: {results['mean_latency']:.2f}ms")
        print(f"  Std Dev: {results['std_latency']:.2f}ms")
        print(f"  Min: {results['min_latency']:.2f}ms")
        print(f"  Max: {results['max_latency']:.2f}ms")
        print(f"  P95: {results['p95_latency']:.2f}ms")
        print(f"  P99: {results['p99_latency']:.2f}ms")
        print()
        
        return results


def benchmark_dynamic_convolution_original_style(input_shape=(3, 224, 224), output_channels_list=[64, 128, 256], **kwargs):
    """
    Replicate the original D_Ptflops.py benchmarking exactly.
    
    Args:
        input_shape: Input shape (C, H, W)
        output_channels_list: List of output channel configurations to test
        **kwargs: Additional arguments for DynamicConv2D
    """
    print("=" * 60)
    print("DYNAMIC CONVOLUTION ANALYSIS")
    print("=" * 60)
    
    for out_channels in output_channels_list:
        print(f"\nBenchmarking with output channels = {out_channels} ({'Small' if out_channels == 64 else 'Medium' if out_channels == 128 else 'Large'} Output):")
        
        model = DynamicConv2D(
            in_channels=input_shape[0], 
            out_channels=out_channels,
            kernel_size=kwargs.get('kernel_size', 3),
            stride=kwargs.get('stride', 1),
            padding=kwargs.get('padding', 1)
        )
        
        model.benchmark_and_print(input_shape, output_channels=out_channels)
