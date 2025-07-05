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
      - Candidate Kernel Weights (αᴄ): weights over candidate depthwise kernels, shape [B, num_candidates, 1, 1, 1]
    
    The base candidate kernels are stored in a bank of shape 
    [num_candidates, C_in, 1, K, K]. They are aggregated (via a weighted sum using αᴄ) and then 
    modulated elementwise by the other attentions to form the final dynamic depthwise kernel.
    
    This dynamic kernel is applied in a depthwise convolution (groups = C_in), 
    and its output is then passed through a fixed pointwise convolution.
    
    An additional output channel attention is applied to the pointwise output.
    
    This implementation includes all the sophisticated analysis capabilities
    from the original DSODConv.py implementation.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, reduction_ratio=4, num_candidates=6):
        super(ODConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # assume square kernel
        self.stride = stride
        self.num_candidates = num_candidates
        self.reduction_ratio = reduction_ratio
        
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
    
    def calculate_flops_and_params(self, input_shape):
        """
        Calculate FLOPs and parameters for ODConv with comprehensive breakdown.
        Exactly replicating the original DSODConv.py calculation methodology.
        
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
        
        # Candidate kernel bank parameters:
        dw_bank_params = self.num_candidates * self.in_channels * (self.kernel_size * self.kernel_size)
        # Pointwise conv parameters:
        pw_params = self.in_channels * self.out_channels + self.out_channels
        # Attention branch parameters (detailed breakdown):
        fc_spatial_params = self.in_channels * (self.kernel_size * self.kernel_size)
        fc_in_params = self.in_channels * self.in_channels
        fc_kernel_params = self.in_channels * (self.in_channels * self.kernel_size * self.kernel_size)
        fc_candidate_params = self.in_channels * self.num_candidates + self.num_candidates
        fc_out_params = self.in_channels * self.out_channels
        attn_params = fc_spatial_params + fc_in_params + fc_kernel_params + fc_candidate_params + fc_out_params
        total_params = dw_bank_params + pw_params + attn_params

        # Compute output spatial dimensions (assuming padding = kernel_size//2)
        H_out = (input_shape_hwc[0] + 2*(self.kernel_size//2) - self.kernel_size) // self.stride + 1
        W_out = (input_shape_hwc[1] + 2*(self.kernel_size//2) - self.kernel_size) // self.stride + 1

        # FLOPs estimation:
        # Depthwise conv FLOPs:
        dw_mults = (self.kernel_size * self.kernel_size * self.in_channels) * H_out * W_out
        # Pointwise conv FLOPs:
        pw_mults = (self.in_channels * self.out_channels) * H_out * W_out
        # Convolution additions:
        dw_adds = ((self.kernel_size * self.kernel_size * self.in_channels) - 1) * H_out * W_out
        pw_adds = ((self.in_channels * self.out_channels) - 1) * H_out * W_out
        conv_adds = dw_adds + pw_adds
        # Divisions:
        divs = H_out * W_out * self.out_channels

        # Dynamic branch FLOPs (detailed estimation):
        gap_flops = self.in_channels * (input_shape_hwc[0] * input_shape_hwc[1])
        fc_spatial_flops = 2 * self.in_channels * (self.kernel_size * self.kernel_size)
        fc_in_flops = 2 * self.in_channels * self.in_channels
        fc_kernel_flops = 2 * self.in_channels * (self.in_channels * self.kernel_size * self.kernel_size)
        fc_candidate_flops = 2 * self.in_channels * self.num_candidates
        fc_out_flops = 2 * self.in_channels * self.out_channels
        fc_total_flops = fc_spatial_flops + fc_in_flops + fc_kernel_flops + fc_candidate_flops + fc_out_flops
        
        # Attention application FLOPs
        softmax_flops = 2 * self.num_candidates
        sigmoid_flops = 2 * (self.kernel_size * self.kernel_size + self.in_channels + 
                           self.in_channels * self.kernel_size * self.kernel_size + self.out_channels)
        
        # Kernel aggregation and modulation FLOPs
        kernel_elements = self.in_channels * (self.kernel_size * self.kernel_size)
        candidate_aggregation_flops = (self.num_candidates + (self.num_candidates - 1)) * kernel_elements
        attention_modulation_flops = 4 * kernel_elements  # 4 attention types applied elementwise
        
        dynamic_flops = (gap_flops + fc_total_flops + softmax_flops + sigmoid_flops + 
                        candidate_aggregation_flops + attention_modulation_flops)

        total_mults = dw_mults + pw_mults + dynamic_flops
        total_flops = total_mults + conv_adds + divs
        output_shape = (H_out, W_out, self.out_channels)
        
        return {
            'total_params': total_params,
            'total_flops': total_flops,
            'dw_bank_params': dw_bank_params,
            'pw_params': pw_params,
            'attn_params': attn_params,
            'fc_spatial_params': fc_spatial_params,
            'fc_in_params': fc_in_params,
            'fc_kernel_params': fc_kernel_params,
            'fc_candidate_params': fc_candidate_params,
            'fc_out_params': fc_out_params,
            'dw_mults': dw_mults,
            'pw_mults': pw_mults,
            'dynamic_flops': dynamic_flops,
            'conv_adds': conv_adds,
            'divisions': divs,
            'output_shape': output_shape,
            'gap_flops': gap_flops,
            'fc_total_flops': fc_total_flops,
            'softmax_flops': softmax_flops,
            'sigmoid_flops': sigmoid_flops,
            'candidate_aggregation_flops': candidate_aggregation_flops,
            'attention_modulation_flops': attention_modulation_flops
        }
    
    def benchmark(self, input_tensor, num_runs=100, warmup_runs=5, detailed=True):
        """
        Comprehensive benchmarking with detailed latency statistics.
        Exactly replicating the original DSODConv.py benchmarking methodology.
        
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
        Print detailed analysis in the original DSODConv.py style.
        
        Args:
            input_shape: Input tensor shape for analysis
            style: Output style ("original", "summary", or "comprehensive")
        """
        analysis = self.calculate_flops_and_params(input_shape)
        
        if style == "original":
            print("\nDetailed Analysis for ODConv with Depthwise Separable Decomposition and Dynamic Attention:")
            print(f"  Kernel Size: {self.kernel_size}")
            print(f"  Stride: {self.stride}")
            print(f"  Num Candidates: {self.num_candidates}")
            print(f"  Output Shape: {analysis['output_shape']}")
            print(f"  Parameters: {analysis['total_params'] / 1e3:.2f}K")
            print(f"  Total FLOPs: {analysis['total_flops'] / 1e6:.2f}M")
            print(f"    - Depthwise Mults: {analysis['dw_mults'] / 1e6:.2f}M")
            print(f"    - Pointwise Mults: {analysis['pw_mults'] / 1e6:.2f}M")
            print(f"    - Dynamic Branch FLOPs: {analysis['dynamic_flops'] / 1e6:.2f}M")
            print(f"    - Convolution Adds: {analysis['conv_adds'] / 1e6:.2f}M")
            print(f"    - Divisions: {analysis['divisions'] / 1e6:.2f}M")
            print("\nParameter Breakdown:")
            print(f"  - Depthwise Kernel Bank: {analysis['dw_bank_params']}")
            print(f"  - Pointwise Conv: {analysis['pw_params']}")
            print(f"  - Attention Branches: {analysis['attn_params']}")
        
        elif style == "summary":
            print(f"ODConv2D: {analysis['total_params']/1e3:.1f}K params, {analysis['total_flops']/1e6:.1f}M FLOPs")
        
        elif style == "comprehensive":
            self.print_detailed_analysis(input_shape, "original")
            print("\nAttention Branch Breakdown:")
            print(f"  - Spatial FC: {analysis['fc_spatial_params']} params")
            print(f"  - Input Channel FC: {analysis['fc_in_params']} params")
            print(f"  - Kernel Weight FC: {analysis['fc_kernel_params']} params")
            print(f"  - Candidate FC: {analysis['fc_candidate_params']} params")
            print(f"  - Output Channel FC: {analysis['fc_out_params']} params")
            print("\nDynamic Branch FLOP Breakdown:")
            print(f"  - GAP FLOPs: {analysis['gap_flops'] / 1e6:.2f}M")
            print(f"  - FC Total FLOPs: {analysis['fc_total_flops'] / 1e6:.2f}M")
            print(f"  - Softmax FLOPs: {analysis['softmax_flops'] / 1e6:.2f}M")
            print(f"  - Sigmoid FLOPs: {analysis['sigmoid_flops'] / 1e6:.2f}M")
            print(f"  - Candidate Aggregation FLOPs: {analysis['candidate_aggregation_flops'] / 1e6:.2f}M")
            print(f"  - Attention Modulation FLOPs: {analysis['attention_modulation_flops'] / 1e6:.2f}M")
    
    def benchmark_and_print(self, input_shape, num_runs=100, output_channels=None):
        """
        Run benchmark and print results in original DSODConv.py style.
        
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
        
        print("\nBenchmarking Results for ODConv with Depthwise Separable Decomposition and Dynamic Attention:")
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


def benchmark_odconv_original_style(input_shape=(224, 224, 3), output_channels_list=[64, 128, 256], **kwargs):
    """
    Replicate the original DSODConv.py benchmarking exactly.
    
    Args:
        input_shape: Input shape (H, W, C)
        output_channels_list: List of output channel configurations to test
        **kwargs: Additional arguments for ODConv2D
    """
    print("=" * 80)
    print("OMNI-DIMENSIONAL DYNAMIC CONVOLUTION ANALYSIS")
    print("=" * 80)
    
    for out_channels in output_channels_list:
        print(f"\nBenchmarking with output channels = {out_channels}:")
        
        model = ODConv2D(
            in_channels=input_shape[2], 
            out_channels=out_channels,
            kernel_size=kwargs.get('kernel_size', 3),
            stride=kwargs.get('stride', 2),
            reduction_ratio=kwargs.get('reduction_ratio', 4),
            num_candidates=kwargs.get('num_candidates', 6)
        )
        
        model.benchmark_and_print(input_shape, output_channels=out_channels)
