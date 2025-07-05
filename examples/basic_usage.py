"""
Enhanced example usage of the conv-benchmarks library.
This demonstrates both the modern API and the sophisticated analysis capabilities.
"""

import torch
from conv_benchmarks import ConvolutionBenchmark
from conv_benchmarks.layers import ODConv2D, KWDSConv2D, DeformableConv2D

def basic_usage_example():
    """Basic usage example with modern API."""
    print("🚀 BASIC USAGE EXAMPLE")
    print("=" * 50)
    
    # Create benchmark instance
    benchmark = ConvolutionBenchmark(device="cpu", num_runs=50)
    
    # Define test configuration
    input_shape = (224, 224, 3)  # (H, W, C)
    output_channels = 64
    kernel_size = 3
    stride = 2
    
    print(f"Benchmarking convolutions with:")
    print(f"  Input shape: {input_shape}")
    print(f"  Output channels: {output_channels}")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Stride: {stride}")
    print("="*50)
    
    # Run comparison
    results = benchmark.compare_all(
        input_shape=input_shape,
        output_channels=output_channels,
        kernel_size=kernel_size,
        stride=stride
    )
    
    # Print results in summary format
    print("\n📊 SUMMARY COMPARISON:")
    benchmark.print_comparison(results, style="summary")

def sophisticated_analysis_example():
    """Demonstrate sophisticated analysis capabilities."""
    print("\n\n🔬 SOPHISTICATED ANALYSIS EXAMPLE")
    print("=" * 50)
    
    # Create benchmark instance
    benchmark = ConvolutionBenchmark(device="cpu", num_runs=30)
    
    input_shape = (224, 224, 3)
    output_channels = 128
    
    print("Individual Layer Analysis with Detailed Breakdowns:\n")
    
    # 1. ODConv Analysis
    print("1. ODConv (Omni-Dimensional Dynamic Convolution):")
    print("-" * 45)
    
    odconv_result = benchmark.benchmark_single(
        ODConv2D, input_shape, output_channels, kernel_size=3, stride=2,
        num_candidates=4  # ODConv specific parameter
    )
    
    # Access the comprehensive analysis correctly
    analysis = odconv_result['comprehensive_analysis']
    manual = analysis.get('manual_calculation')
    
    if manual:
        print(f"  Parameters: {manual['total_params'] / 1e3:.2f}K")
        print(f"  Total FLOPs: {manual['total_flops'] / 1e6:.2f}M")
        print(f"    - Depthwise Mults: {manual['depthwise_mults'] / 1e6:.2f}M")
        print(f"    - Pointwise Mults: {manual['pointwise_mults'] / 1e6:.2f}M")
        print(f"    - Dynamic Branch FLOPs: {manual['dynamic_flops'] / 1e6:.2f}M")
    
    print(f"  Mean Latency: {odconv_result['latency_mean']:.2f}ms")
    print(f"  P95 Latency: {odconv_result['latency_p95']:.2f}ms")
    
    # 2. KWDS Analysis
    print("\n2. KWDS (Kernel Warehouse Depthwise Separable):")
    print("-" * 50)
    
    kwds_result = benchmark.benchmark_single(
        KWDSConv2D, input_shape, output_channels, kernel_size=3, stride=2,
        num_kernels=4  # KWDS specific parameter
    )
    
    analysis = kwds_result['comprehensive_analysis']
    manual = analysis.get('manual_calculation')
    
    if manual:
        print(f"  Parameters: {manual['total_params'] / 1e3:.2f}K")
        print(f"  Total FLOPs: {manual['total_flops'] / 1e6:.2f}M")
        print(f"    - Depthwise Mults: {manual['depthwise_mults'] / 1e6:.2f}M")
        print(f"    - Pointwise Mults: {manual['pointwise_mults'] / 1e6:.2f}M")
        print(f"    - Dynamic Branch FLOPs: {manual['dynamic_flops'] / 1e6:.2f}M")
    
    print(f"  Mean Latency: {kwds_result['latency_mean']:.2f}ms")
    print(f"  P95 Latency: {kwds_result['latency_p95']:.2f}ms")

def direct_layer_usage_example():
    """Demonstrate direct layer usage with built-in analysis."""
    print("\n\n🎯 DIRECT LAYER USAGE EXAMPLE")
    print("=" * 50)
    
    # Create input tensor
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # 1. KWDSConv2D with built-in analysis
    print("1. KWDSConv2D Built-in Analysis:")
    print("-" * 35)
    
    kwds_layer = KWDSConv2D(in_channels=3, out_channels=64, kernel_size=3, stride=2, num_kernels=4)
    
    # Use built-in benchmarking
    results = kwds_layer.benchmark(input_tensor, num_runs=50, detailed=True)
    
    # Print sophisticated analysis exactly like original DSKW.py
    print(f"  Output Shape: {results['output_shape']}")
    print(f"  Parameters: {results['total_params'] / 1e3:.2f}K")
    print(f"  Total FLOPs: {results['total_flops'] / 1e6:.2f}M")
    print(f"    - Depthwise Mults: {results['dw_mults'] / 1e6:.2f}M")
    print(f"    - Pointwise Mults: {results['pw_mults'] / 1e6:.2f}M")
    print(f"    - Dynamic Branch FLOPs: {results['dynamic_flops'] / 1e6:.2f}M")
    print(f"    - Convolution Adds: {results['conv_adds'] / 1e6:.2f}M")
    print(f"    - Divisions: {results['divisions'] / 1e6:.2f}M")
    print(f"  Mean Latency: {results['mean_latency']:.2f}ms")
    print(f"  P95/P99: {results['p95_latency']:.2f}ms / {results['p99_latency']:.2f}ms")
    
    # 2. ODConv2D with built-in analysis
    print("\n2. ODConv2D Built-in Analysis:")
    print("-" * 30)
    
    odconv_layer = ODConv2D(in_channels=3, out_channels=64, kernel_size=3, stride=2, num_candidates=6)
    
    results = odconv_layer.benchmark(input_tensor, num_runs=50, detailed=True)
    
    print(f"  Output Shape: {results['output_shape']}")
    print(f"  Parameters: {results['total_params'] / 1e3:.2f}K")
    print(f"  Total FLOPs: {results['total_flops'] / 1e6:.2f}M")
    print(f"    - Depthwise Mults: {results['dw_mults'] / 1e6:.2f}M")
    print(f"    - Pointwise Mults: {results['pw_mults'] / 1e6:.2f}M")
    print(f"    - Dynamic Branch FLOPs: {results['dynamic_flops'] / 1e6:.2f}M")
    print(f"  Mean Latency: {results['mean_latency']:.2f}ms")
    print(f"  P95/P99: {results['p95_latency']:.2f}ms / {results['p99_latency']:.2f}ms")

def original_style_replication_example():
    """Demonstrate exact original script replication."""
    print("\n\n🎬 ORIGINAL SCRIPT REPLICATION EXAMPLE")
    print("=" * 50)
    
    from conv_benchmarks import replicate_dskw, replicate_dsodconv, replicate_dcnv1
    
    print("1. Replicating Original DSKW.py:")
    print("-" * 35)
    try:
        replicate_dskw()
    except Exception as e:
        print(f"Note: {e}")
        print("Using direct function instead...")
        from conv_benchmarks.layers.kernel_warehouse import benchmark_kwds_conv_original_style
        benchmark_kwds_conv_original_style(output_channels_list=[64])
    
    print("\n2. Replicating Original DSODConv.py:")
    print("-" * 40)
    try:
        replicate_dsodconv() 
    except Exception as e:
        print(f"Note: {e}")
        print("Using direct function instead...")
        from conv_benchmarks.layers.omni_dimensional import benchmark_odconv_original_style
        benchmark_odconv_original_style(output_channels_list=[64])

def comprehensive_comparison_example():
    """Show comprehensive comparison with all details."""
    print("\n\n📈 COMPREHENSIVE COMPARISON EXAMPLE") 
    print("=" * 50)
    
    benchmark = ConvolutionBenchmark(device="cpu", num_runs=30)
    
    # Run full comparison
    results = benchmark.compare_all(
        input_shape=(224, 224, 3),
        output_channels=64,
        kernel_size=3,
        stride=2
    )
    
    # Print comprehensive results showing all sophisticated analysis
    print("COMPREHENSIVE ANALYSIS WITH ORIGINAL DEPTH:")
    benchmark.print_comparison(results, style="comprehensive")

def advanced_configuration_example():
    """Show advanced configuration options."""
    print("\n\n⚙️ ADVANCED CONFIGURATION EXAMPLE")
    print("=" * 50)
    
    # Custom configurations for different layers
    kwds_config = {
        'num_kernels': 8,  # More kernel candidates
        'kernel_size': 5,  # Larger kernel
        'stride': 1
    }
    
    odconv_config = {
        'num_candidates': 8,  # More candidate kernels
        'reduction_ratio': 2,  # Different reduction
        'kernel_size': 3,
        'stride': 2
    }
    
    print("1. Advanced KWDS Configuration:")
    print(f"   - Num Kernels: {kwds_config['num_kernels']}")
    print(f"   - Kernel Size: {kwds_config['kernel_size']}")
    
    kwds_layer = KWDSConv2D(
        in_channels=3, 
        out_channels=128, 
        **kwds_config
    )
    
    # Print detailed analysis
    kwds_layer.print_detailed_analysis((224, 224, 3), style="comprehensive")
    
    print("\n2. Advanced ODConv Configuration:")
    print(f"   - Num Candidates: {odconv_config['num_candidates']}")
    print(f"   - Reduction Ratio: {odconv_config['reduction_ratio']}")
    
    odconv_layer = ODConv2D(
        in_channels=3,
        out_channels=128,
        **odconv_config
    )
    
    # Print detailed analysis  
    odconv_layer.print_detailed_analysis((224, 224, 3), style="comprehensive")

def main():
    """Run all examples to demonstrate library capabilities."""
    print("🔬 CONV-BENCHMARKS LIBRARY DEMONSTRATION")
    print("=" * 60)
    print("This example showcases the sophisticated analysis capabilities")
    print("preserved and enhanced from your original implementations.")
    print("=" * 60)
    
    # Run all examples
    basic_usage_example()
    sophisticated_analysis_example()  
    direct_layer_usage_example()
    original_style_replication_example()
    comprehensive_comparison_example()
    advanced_configuration_example()
    
    print("\n\n✅ DEMONSTRATION COMPLETE")
    print("=" * 50)
    print("The library successfully preserves and enhances all the")
    print("sophisticated analysis capabilities from your original work!")

if __name__ == "__main__":
    main()
