"""
Advanced example showing detailed FLOP analysis and original-style benchmarking.
This replicates the detailed output from your original scripts.
"""

from conv_benchmarks import ConvolutionBenchmark
from conv_benchmarks.layers import KWDSConv2D, ODConv2D
import torch

def replicate_original_benchmarking():
    """
    Replicate the detailed benchmarking style from your original scripts.
    """
    print("🔬 Advanced Conv Benchmarks - Detailed Analysis")
    print("=" * 60)
    
    # Create benchmark instance
    benchmark = ConvolutionBenchmark(device="cpu", num_runs=100, use_ptflops=True)
    
    # Test configurations from your original scripts
    input_shape = (224, 224, 3)
    output_configs = [
        (64, "Small Output"),
        (128, "Medium Output"), 
        (256, "Large Output")
    ]
    
    for output_channels, config_name in output_configs:
        print(f"\n🎯 Benchmarking with output channels = {output_channels} ({config_name}):")
        print("-" * 50)
        
        # Run comparison with detailed analysis
        results = benchmark.compare_all(
            input_shape=input_shape,
            output_channels=output_channels,
            kernel_size=3,
            stride=2,  # Using stride=2 like your originals
            num_kernels=4,  # For KWDS
            num_candidates=6,  # For ODConv
        )
        
        # Print detailed breakdown
        benchmark.print_comparison(results, detailed=True)

def individual_layer_analysis():
    """
    Individual layer analysis similar to your original benchmark functions.
    """
    print("\n" + "=" * 60)
    print("🔍 Individual Layer Analysis")
    print("=" * 60)
    
    benchmark = ConvolutionBenchmark(device="cpu", num_runs=100)
    input_shape = (224, 224, 3)
    
    # Test KWDS Conv (replicating DSKW.py style)
    print("\n📈 Kernel Warehouse Depthwise Separable Conv Analysis:")
    print("-" * 50)
    
    kwds_result = benchmark.benchmark_single(
        KWDSConv2D,
        input_shape=input_shape,
        output_channels=64,
        kernel_size=3,
        stride=2,
        num_kernels=4
    )
    
    flop_details = kwds_result["flop_details"]
    print(f"Parameters: {kwds_result['parameters'] / 1e3:.2f}K")
    print(f"Total FLOPs: {flop_details['total_flops'] / 1e6:.2f}M")
    print(f"  - Depthwise Mults: {flop_details['depthwise_mults'] / 1e6:.2f}M")
    print(f"  - Pointwise Mults: {flop_details['pointwise_mults'] / 1e6:.2f}M")
    print(f"  - Dynamic Branch FLOPs: {flop_details['dynamic_flops'] / 1e6:.2f}M")
    print(f"  - Convolution Adds: {flop_details['conv_adds'] / 1e6:.2f}M")
    print(f"  - Divisions: {flop_details['divisions'] / 1e6:.2f}M")
    print(f"Output Shape: {flop_details['output_shape']}")
    print("Latency Statistics:")
    print(f"  Mean: {kwds_result['latency_mean']:.2f}ms")
    print(f"  Std Dev: {kwds_result['latency_std']:.2f}ms")
    print(f"  Min: {kwds_result['latency_min']:.2f}ms | Max: {kwds_result['latency_max']:.2f}ms")
    print(f"  P95: {kwds_result['latency_p95']:.2f}ms | P99: {kwds_result['latency_p99']:.2f}ms")
    
    # Test ODConv (replicating DSODConv.py style)
    print("\n📈 Omni-Dimensional Dynamic Conv Analysis:")
    print("-" * 50)
    
    odconv_result = benchmark.benchmark_single(
        ODConv2D,
        input_shape=input_shape,
        output_channels=64,
        kernel_size=3,
        stride=2,
        num_candidates=6
    )
    
    flop_details = odconv_result["flop_details"]
    print(f"Parameters: {odconv_result['parameters'] / 1e3:.2f}K")
    print(f"Total FLOPs: {flop_details['total_flops'] / 1e6:.2f}M")
    print(f"  - Depthwise Mults: {flop_details['depthwise_mults'] / 1e6:.2f}M")
    print(f"  - Pointwise Mults: {flop_details['pointwise_mults'] / 1e6:.2f}M")
    print(f"  - Dynamic Branch FLOPs: {flop_details['dynamic_flops'] / 1e6:.2f}M")
    print(f"  - Convolution Adds: {flop_details['conv_adds'] / 1e6:.2f}M")
    print(f"  - Divisions: {flop_details['divisions'] / 1e6:.2f}M")
    print(f"Output Shape: {flop_details['output_shape']}")
    print("Latency Statistics:")
    print(f"  Mean: {odconv_result['latency_mean']:.2f}ms")
    print(f"  Std Dev: {odconv_result['latency_std']:.2f}ms")
    print(f"  Min: {odconv_result['latency_min']:.2f}ms | Max: {odconv_result['latency_max']:.2f}ms")
    print(f"  P95: {odconv_result['latency_p95']:.2f}ms | P99: {odconv_result['latency_p99']:.2f}ms")

def efficiency_analysis():
    """
    Efficiency analysis comparing parameter count vs performance.
    """
    print("\n" + "=" * 60)
    print("⚡ Efficiency Analysis")
    print("=" * 60)
    
    benchmark = ConvolutionBenchmark(device="cpu", num_runs=50)
    input_shape = (224, 224, 3)
    
    results = benchmark.compare_all(
        input_shape=input_shape,
        output_channels=128,
        kernel_size=3,
        stride=2
    )
    
    # Calculate efficiency metrics
    print("\n📊 Efficiency Metrics:")
    print("-" * 30)
    
    efficiency_data = []
    for name, result in results.items():
        if result is None:
            continue
            
        params = result["parameters"]
        flops = result["flop_details"]["total_flops"]
        latency = result["latency_mean"]
        
        # Efficiency metrics
        flops_per_param = flops / params
        throughput = 1000 / latency  # inferences per second
        
        efficiency_data.append({
            "name": name,
            "params": params,
            "flops": flops,
            "latency": latency,
            "flops_per_param": flops_per_param,
            "throughput": throughput
        })
    
    # Sort by efficiency (lowest latency first)
    efficiency_data.sort(key=lambda x: x["latency"])
    
    print(f"{'Convolution':<20} {'Params':<10} {'FLOPs(M)':<10} {'Latency':<10} {'Throughput':<12}")
    print("-" * 70)
    for data in efficiency_data:
        print(f"{data['name']:<20} {data['params']/1e3:.1f}K     {data['flops']/1e6:.1f}M      "
              f"{data['latency']:.2f}ms    {data['throughput']:.1f} inf/s")
    
    print(f"\n🏆 Most Efficient (by latency): {efficiency_data[0]['name']}")
    print(f"🔥 Highest Throughput: {max(efficiency_data, key=lambda x: x['throughput'])['name']}")
    print(f"💾 Fewest Parameters: {min(efficiency_data, key=lambda x: x['params'])['name']}")

def save_detailed_results():
    """
    Save results to JSON for further analysis.
    """
    print("\n" + "=" * 60)
    print("💾 Saving Detailed Results")
    print("=" * 60)
    
    benchmark = ConvolutionBenchmark(device="cpu", num_runs=20)  # Faster for demo
    
    results = benchmark.compare_all(
        input_shape=(224, 224, 3),
        output_channels=64,
        kernel_size=3,
        stride=2
    )
    
    # Save results
    benchmark.save_results(results, "conv_benchmark_results.json")
    print("✅ Detailed results saved to 'conv_benchmark_results.json'")
    print("📈 You can now analyze the results with external tools or load them back for comparison.")

def main():
    """Main function demonstrating all capabilities."""
    # Run all analyses
    replicate_original_benchmarking()
    individual_layer_analysis()
    efficiency_analysis()
    save_detailed_results()
    
    print("\n" + "=" * 60)
    print("🎉 Advanced Benchmarking Complete!")
    print("=" * 60)
    print("\n✨ Your conv-benchmarks library successfully replicates and enhances")
    print("   the detailed analysis capabilities of your original scripts!")

if __name__ == "__main__":
    main()
