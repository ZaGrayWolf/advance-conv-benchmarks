"""
Enhanced ConvolutionBenchmark with exact original script integration.
This provides both the unified API and exact original script replication.
"""
import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from tabulate import tabulate

class ConvolutionBenchmark:
    """
    Enhanced benchmarking suite that provides both unified API and exact original script replication.
    This ensures maximum depth and comprehensiveness matching your original implementations.
    """
    
    def __init__(self, device: str = "cpu", num_runs: int = 100, use_ptflops: bool = True):
        """
        Initialize the benchmark.
        
        Args:
            device: Device to run benchmarks on ("cpu" or "cuda")
            num_runs: Number of iterations for latency measurement
            use_ptflops: Whether to use ptflops library for accurate FLOP counting
        """
        self.device = torch.device(device)
        self.num_runs = num_runs
        self.use_ptflops = use_ptflops
        
        # Import layers here to avoid circular imports
        from ..layers import (
            TraditionalConv2D,
            DepthwiseSeparableConv, 
            DeformableConv2D,
            DynamicConv2D,
            KWDSConv2D,
            ODConv2D,
        )
        
        self.conv_types = {
            "Traditional": TraditionalConv2D,
            "Depthwise Separable": DepthwiseSeparableConv,
            "Deformable v1": DeformableConv2D, 
            "Dynamic": DynamicConv2D,
            "Kernel Warehouse": KWDSConv2D,
            "ODConv": ODConv2D,
        }
    
    def replicate_original_script(self, script_name: str, input_shape: Optional[Tuple[int, int, int]] = None):
        """
        Run EXACT replication of an original script.
        This preserves every detail of your original implementations.
        
        Args:
            script_name: Name of original script ("DSKW", "DSODConv", "DCNv1", etc.)
            input_shape: Input shape (optional, uses default if not provided)
        """
        print(f"🎯 Running EXACT replication of {script_name}.py")
        print("=" * 60)
        print("This output matches your original script exactly:")
        print("-" * 60)
        
        # Import here to avoid circular imports
        from ..original_scripts import run_original_script
        
        # Run the exact original script
        run_original_script(script_name)
        
        print("-" * 60)
        print(f"✅ Completed exact replication of {script_name}.py")
    
    def detailed_analysis(self, conv_class, input_shape, output_channels, **kwargs):
        """
        Return the full sophisticated breakdown like original scripts.
        This method exposes all the advanced analysis capabilities directly.
        
        Args:
            conv_class: Convolution class to analyze
            input_shape: Input tensor shape (H, W, C)
            output_channels: Number of output channels
            **kwargs: Additional arguments for the convolution layer
            
        Returns:
            Dict containing comprehensive analysis with original-level detail
        """
        from ..layers import KWDSConv2D, ODConv2D, DepthwiseSeparableConv, DeformableConv2D
        
        in_channels = input_shape[2]
        
        # Create model based on type
        if conv_class == KWDSConv2D:
            model = conv_class(in_channels, output_channels, **kwargs)
            conv_type = "kernel_warehouse"
        elif conv_class == ODConv2D:
            model = conv_class(in_channels, output_channels, **kwargs)
            conv_type = "omni_dimensional"
        elif conv_class == DepthwiseSeparableConv:
            model = conv_class(in_channels, output_channels, **kwargs)
            conv_type = "depthwise_separable"
        elif conv_class == DeformableConv2D:
            model = conv_class(in_channels, output_channels, **kwargs)
            conv_type = "deformable"
        else:
            model = conv_class(in_channels, output_channels, **kwargs)
            conv_type = "standard"
            
        model = model.to(self.device).eval()
        
        # Create input tensor
        input_tensor = torch.randn(1, in_channels, input_shape[0], input_shape[1]).to(self.device)
        
        # Use built-in sophisticated analysis from enhanced layers
        detailed_results = model.benchmark(input_tensor, num_runs=self.num_runs, detailed=True)
        
        # Add model information
        detailed_results.update({
            'conv_type': conv_type,
            'model_class': conv_class.__name__,
            'input_shape': input_shape,
            'output_channels': output_channels,
            'device': str(self.device)
        })
        
        return detailed_results
    
    def print_original_style_detailed(self, conv_class, input_shape, output_channels, **kwargs):
        """
        Print analysis in the exact original script style with full sophisticated breakdown.
        
        Args:
            conv_class: Convolution class to analyze
            input_shape: Input tensor shape (H, W, C)
            output_channels: Number of output channels
            **kwargs: Additional arguments for the convolution layer
        """
        from ..layers import KWDSConv2D, ODConv2D, DeformableConv2D, TraditionalConv2D, DepthwiseSeparableConv, DynamicConv2D
        
        print("\n1. KERNEL WAREHOUSE DEPTHWISE SEPARABLE:")
        self._benchmark_kwds_conv_original_style(input_shape, output_channels_list)
        
        print("\n2. OMNI-DIMENSIONAL DYNAMIC CONVOLUTION:")
        self._benchmark_odconv_original_style(input_shape, output_channels_list)
        
        print("\n3. DEFORMABLE CONVOLUTION v1:")
        self._benchmark_deformable_convolution_original_style(input_shape, output_channels_list)
        
        print("\n4. TRADITIONAL CONVOLUTION:")
        # Convert to (C, H, W) format for traditional
        trad_input_shape = (input_shape[2], input_shape[0], input_shape[1])
        self._benchmark_traditional_convolution_original_style(trad_input_shape, output_channels_list)
        
        print("\n5. DEPTHWISE SEPARABLE CONVOLUTION:")
        self._benchmark_depthwise_separable_original_style(input_shape, output_channels_list)
        
        print("\n6. DYNAMIC CONVOLUTION:")
        dynamic_input_shape = (input_shape[2], input_shape[0], input_shape[1])
        self._benchmark_dynamic_convolution_original_style(dynamic_input_shape, output_channels_list)
        
        print("\n✅ COMPREHENSIVE ORIGINAL-STYLE BENCHMARKING COMPLETE")
        print("=" * 80)
    
    def _benchmark_kwds_conv_original_style(self, input_shape=(224, 224, 3), output_channels_list=[64, 128, 256], **kwargs):
        """Replicate the original DSKW.py benchmarking exactly."""
        from ..layers import KWDSConv2D
        
        print("=" * 60)
        print("KERNEL WAREHOUSE DEPTHWISE SEPARABLE CONVOLUTION ANALYSIS")
        print("=" * 60)
        
        for out_channels in output_channels_list:
            print(f"\nBenchmarking with output channels = {out_channels}:")
            config = {
                'kernel_size': kwargs.get('kernel_size', 3),
                'stride': kwargs.get('stride', 2),
                'num_kernels': kwargs.get('num_kernels', 4)
            }
            self.print_original_style_detailed(KWDSConv2D, input_shape, out_channels, **config)
    
    def _benchmark_odconv_original_style(self, input_shape=(224, 224, 3), output_channels_list=[64, 128, 256], **kwargs):
        """Replicate the original DSODConv.py benchmarking exactly."""
        from ..layers import ODConv2D
        
        print("=" * 60)
        print("OMNI-DIMENSIONAL DYNAMIC CONVOLUTION ANALYSIS")
        print("=" * 60)
        
        for out_channels in output_channels_list:
            print(f"\nBenchmarking with output channels = {out_channels}:")
            config = {
                'kernel_size': kwargs.get('kernel_size', 3),
                'stride': kwargs.get('stride', 2),
                'num_candidates': kwargs.get('num_candidates', 6)
            }
            self.print_original_style_detailed(ODConv2D, input_shape, out_channels, **config)
    
    def _benchmark_deformable_convolution_original_style(self, input_shape=(224, 224, 3), output_channels_list=[64, 128, 256], **kwargs):
        """Replicate the original DCNv1.py benchmarking exactly."""
        from ..layers import DeformableConv2D
        
        print("=" * 60)
        print("DEFORMABLE CONVOLUTION v1 ANALYSIS")
        print("=" * 60)
        
        for out_channels in output_channels_list:
            print(f"\nBenchmarking with output channels = {out_channels}:")
            config = {
                'kernel_size': kwargs.get('kernel_size', 3),
                'stride': kwargs.get('stride', 1),
                'padding': kwargs.get('padding', 1)
            }
            self.print_original_style_detailed(DeformableConv2D, input_shape, out_channels, **config)
    
    def _benchmark_traditional_convolution_original_style(self, input_shape=(3, 224, 224), output_channels_list=[64, 128, 256], **kwargs):
        """Replicate the original trad_convo.py benchmarking exactly."""
        from ..layers import TraditionalConv2D
        
        print("=" * 60)
        print("TRADITIONAL CONVOLUTION ANALYSIS")
        print("=" * 60)
        
        # Convert input shape back to (H, W, C) for consistency
        if len(input_shape) == 3 and input_shape[0] < input_shape[1]:
            converted_shape = (input_shape[1], input_shape[2], input_shape[0])
        else:
            converted_shape = input_shape
        
        for out_channels in output_channels_list:
            print(f"\nBenchmarking with output channels = {out_channels}:")
            config = {
                'kernel_size': kwargs.get('kernel_size', 3),
                'stride': kwargs.get('stride', 2),
                'padding': kwargs.get('padding', 1)
            }
            self.print_original_style_detailed(TraditionalConv2D, converted_shape, out_channels, **config)
    
    def _benchmark_depthwise_separable_original_style(self, input_shape=(224, 224, 3), output_channels_list=[64, 128, 256], **kwargs):
        """Replicate the original DS_Ptflops.py benchmarking exactly."""
        from ..layers import DepthwiseSeparableConv
        
        print("=" * 60)
        print("DEPTHWISE SEPARABLE CONVOLUTION ANALYSIS")
        print("=" * 60)
        
        for out_channels in output_channels_list:
            print(f"\nBenchmarking with output channels = {out_channels}:")
            config = {
                'kernel_size': kwargs.get('kernel_size', 3),
                'stride': kwargs.get('stride', 2),
                'padding': kwargs.get('padding', 1)
            }
            self.print_original_style_detailed(DepthwiseSeparableConv, input_shape, out_channels, **config)
    
    def _benchmark_dynamic_convolution_original_style(self, input_shape=(3, 224, 224), output_channels_list=[64, 128, 256], **kwargs):
        """Replicate the original D_Ptflops.py benchmarking exactly."""
        from ..layers import DynamicConv2D
        
        print("=" * 60)
        print("DYNAMIC CONVOLUTION ANALYSIS")
        print("=" * 60)
        
        # Convert input shape back to (H, W, C) for consistency
        if len(input_shape) == 3 and input_shape[0] < input_shape[1]:
            converted_shape = (input_shape[1], input_shape[2], input_shape[0])
        else:
            converted_shape = input_shape
        
        for out_channels in output_channels_list:
            print(f"\nBenchmarking with output channels = {out_channels}:")
            config = {
                'kernel_size': kwargs.get('kernel_size', 3),
                'stride': kwargs.get('stride', 2),
                'num_experts': kwargs.get('num_experts', 4)
            }
            self.print_original_style_detailed(DynamicConv2D, converted_shape, out_channels, **config)
    
    def advanced_comparison(self, input_shape=(224, 224, 3), output_channels=64, **layer_configs):
        """Advanced comparison with sophisticated analysis and custom configurations."""
        print("🔬 ADVANCED SOPHISTICATED COMPARISON")
        print("=" * 60)
        
        from ..layers import KWDSConv2D, ODConv2D, DeformableConv2D, TraditionalConv2D, DepthwiseSeparableConv
        
        # Default configurations
        configs = {
            'kwds': layer_configs.get('kwds', {'kernel_size': 3, 'stride': 2, 'num_kernels': 4}),
            'odconv': layer_configs.get('odconv', {'kernel_size': 3, 'stride': 2, 'num_candidates': 6}),
            'deformable': layer_configs.get('deformable', {'kernel_size': 3, 'stride': 1, 'padding': 1}),
            'traditional': layer_configs.get('traditional', {'kernel_size': 3, 'stride': 2}),
            'depthwise': layer_configs.get('depthwise', {'kernel_size': 3, 'stride': 2})
        }
        
        results = {}
        
        print("\nDetailed Layer Analysis:")
        print("-" * 40)
        
        for name, (layer_class, config) in [
            ("KWDS", (KWDSConv2D, configs['kwds'])),
            ("ODConv", (ODConv2D, configs['odconv'])),
            ("Deformable", (DeformableConv2D, configs['deformable'])),
            ("Traditional", (TraditionalConv2D, configs['traditional'])),
            ("Depthwise Sep", (DepthwiseSeparableConv, configs['depthwise']))
        ]:
            print(f"\n{name} Analysis:")
            try:
                result = self.detailed_analysis(layer_class, input_shape, output_channels, **config)
                results[name] = result
                
                params = result.get('total_params', 0)
                flops = result.get('total_flops', 0)
                latency = result.get('mean_latency', 0)
                
                print(f"  Parameters: {params / 1e3:.2f}K")
                print(f"  FLOPs: {flops / 1e6:.2f}M")
                print(f"  Latency: {latency:.2f}ms (±{result.get('std_latency', 0):.2f})")
                print(f"  P95/P99: {result.get('p95_latency', 0):.2f}/{result.get('p99_latency', 0):.2f}ms")
                
                if name == "KWDS" and 'dw_mults' in result:
                    print(f"  - Depthwise Mults: {result['dw_mults'] / 1e6:.2f}M")
                    print(f"  - Pointwise Mults: {result['pw_mults'] / 1e6:.2f}M")
                    print(f"  - Dynamic Branch: {result['dynamic_flops'] / 1e6:.2f}M")
                elif name == "ODConv" and 'dw_mults' in result:
                    print(f"  - 4D Attention Analysis:")
                    print(f"    * Depthwise: {result['dw_mults'] / 1e6:.2f}M FLOPs")
                    print(f"    * Pointwise: {result['pw_mults'] / 1e6:.2f}M FLOPs")
                    print(f"    * Dynamic: {result['dynamic_flops'] / 1e6:.2f}M FLOPs")
                    
            except Exception as e:
                print(f"  Error: {e}")
                results[name] = None
        
        return results
    
    def benchmark_single(self, conv_class: nn.Module, input_shape: Tuple[int, int, int], output_channels: int, 
                        kernel_size: int = 3, stride: int = 1, padding: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Benchmark a single convolution type with comprehensive analysis."""
        from ..utils import get_comprehensive_analysis
        from ..layers import KWDSConv2D, ODConv2D, DepthwiseSeparableConv, DeformableConv2D
        
        in_channels = input_shape[2]
        if padding is None:
            padding = kernel_size // 2
        
        # Create model based on type
        if conv_class == KWDSConv2D:
            model = conv_class(in_channels, output_channels, kernel_size, stride, **kwargs)
            conv_type = "kernel_warehouse"
        elif conv_class == ODConv2D:
            model = conv_class(in_channels, output_channels, kernel_size, stride, **kwargs)
            conv_type = "omni_dimensional"
        elif conv_class == DepthwiseSeparableConv:
            model = conv_class(in_channels, output_channels, kernel_size, stride, padding)
            conv_type = "depthwise_separable"
        elif conv_class == DeformableConv2D:
            model = conv_class(in_channels, output_channels, kernel_size, stride, padding, **kwargs)
            conv_type = "deformable"
        else:
            model = conv_class(in_channels, output_channels, kernel_size, stride, padding)
            conv_type = "standard"
            
        model = model.to(self.device).eval()
        input_tensor = torch.randn(1, in_channels, input_shape[0], input_shape[1]).to(self.device)
        
        # Warm-up
        with torch.no_grad():
            output = model(input_tensor)
        
        # Benchmark latency
        latencies = []
        for _ in range(self.num_runs):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(input_tensor)
                    
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)
        
        latencies = np.array(latencies)
        output_shape = tuple(output.shape[1:])
        
        # Get comprehensive analysis
        analysis_input_shape = (in_channels, input_shape[0], input_shape[1])
        comprehensive_analysis = get_comprehensive_analysis(model, analysis_input_shape, conv_type, **kwargs)
        
        result = {
            "conv_type": conv_type,
            "model_params": sum(p.numel() for p in model.parameters()),
            "output_shape": output_shape,
            "kernel_size": kernel_size,
            "input_shape": input_shape,
            "output_channels": output_channels,
            "stride": stride,
            "padding": padding,
            "latency_mean": np.mean(latencies),
            "latency_std": np.std(latencies),
            "latency_min": np.min(latencies),
            "latency_max": np.max(latencies),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99),
            "comprehensive_analysis": comprehensive_analysis,
        }
        
        return result
    
    def compare_all(self, input_shape: Tuple[int, int, int], output_channels: int, kernel_size: int = 3, 
                   stride: int = 1, padding: Optional[int] = None, **kwargs) -> Dict[str, Dict[str, Any]]:
        """Compare all convolution types with comprehensive analysis."""
        results = {}
        
        for name, conv_class in self.conv_types.items():
            try:
                print(f"Benchmarking {name}...")
                result = self.benchmark_single(conv_class, input_shape, output_channels, kernel_size, stride, padding, **kwargs)
                results[name] = result
            except Exception as e:
                print(f"Failed to benchmark {name}: {e}")
                results[name] = None
        
        return results
    
    def print_original_style_breakdown(self, result: Dict[str, Any], conv_name: str) -> None:
        """Print results in the exact style of your original scripts with full detail breakdown."""
        analysis = result["comprehensive_analysis"]
        manual = analysis.get("manual_calculation")
        ptflops = analysis.get("ptflops_analysis")
        
        if conv_name in ["Kernel Warehouse", "ODConv"]:
            print(f"\nBenchmarking Results for {conv_name}:")
            print(f"  Output Channels: {result['output_channels']}")
            print(f"  Kernel Size: {result['kernel_size']}")
            print(f"  Stride: {result['stride']}")
            
            if manual:
                print(f"  Output Shape: {manual.get('output_shape', result['output_shape'])}")
                print(f"  Parameters: {manual['total_params'] / 1e3:.2f}K")
                print(f"  Total FLOPs: {manual['total_flops'] / 1e6:.2f}M")
                
                if "depthwise_mults" in manual:
                    print(f"    - Depthwise Mults: {manual['depthwise_mults'] / 1e6:.2f}M")
                    print(f"    - Pointwise Mults: {manual['pointwise_mults'] / 1e6:.2f}M")
                    print(f"    - Dynamic Branch FLOPs: {manual['dynamic_flops'] / 1e6:.2f}M")
                    print(f"    - Convolution Adds: {manual['conv_adds'] / 1e6:.2f}M")
                    print(f"    - Divisions: {manual['divisions'] / 1e6:.2f}M")
        
        elif conv_name == "Deformable v1":
            print("Deformable Convolution Layer:")
            if ptflops:
                print(f"Total Parameters: {ptflops['params']} ({ptflops['params_str']})")
                print(f"Estimated FLOPs: {ptflops['flops']} (~{ptflops['flops_str']})")
            elif manual:
                print(f"Total Parameters: {manual['total_params']} ({manual['total_params'] / 1e3:.2f}K)")
                print(f"Estimated FLOPs: {manual['total_flops']} (~{manual['total_flops'] / 1e6:.2f}M)")
        
        else:
            print(f"\n{conv_name}:")
            print(f"Kernel Size: {result['kernel_size']}")
            print(f"Input Shape: {result['input_shape']}")
            print(f"Output Channels: {result['output_channels']}")
            
            if ptflops:
                print(f"Parameters: {ptflops['params_str']}")
                print(f"MACs: {ptflops['macs_str']}")
                print(f"FLOPs: {ptflops['flops_str']}")
            elif manual:
                print(f"Parameters: {manual['total_params'] / 1e3:.2f}K")
                print(f"Total FLOPs: {manual['total_flops'] / 1e6:.2f}M")
        
        # Always show latency statistics
        print("Latency Statistics:")
        print(f"  Mean: {result['latency_mean']:.2f}ms")
        print(f"  Std Dev: {result['latency_std']:.2f}ms")
        print(f"  Min: {result['latency_min']:.2f}ms")
        print(f"  Max: {result['latency_max']:.2f}ms")
        print(f"  P95: {result['latency_p95']:.2f}ms")
        print(f"  P99: {result['latency_p99']:.2f}ms")
        print()
    
    def print_comparison(self, results: Dict[str, Dict[str, Any]], style: str = "original") -> None:
        """Print comparison results in various styles."""
        if style == "original":
            for name, result in results.items():
                if result is not None:
                    self.print_original_style_breakdown(result, name)
        
        elif style == "summary":
            headers = ["Convolution Type", "Parameters", "FLOPs (M)", "Mean Latency (ms)", "P95 Latency (ms)"]
            
            table_data = []
            for name, result in results.items():
                if result is None:
                    continue
                
                analysis = result["comprehensive_analysis"]
                
                if analysis.get("ptflops_analysis"):
                    ptflops = analysis["ptflops_analysis"]
                    params_str = ptflops["params_str"]
                    flops_str = ptflops["flops_str"]
                elif analysis.get("manual_calculation"):
                    manual = analysis["manual_calculation"]
                    params = manual["total_params"]
                    flops = manual["total_flops"]
                    params_str = f"{params / 1e3:.2f}K"
                    flops_str = f"{flops / 1e6:.1f}M"
                else:
                    params_str = "N/A"
                    flops_str = "N/A"
                    
                row = [name, params_str, flops_str, f"{result['latency_mean']:.2f}", f"{result['latency_p95']:.2f}"]
                table_data.append(row)
            
            print("\nConvolution Comparison Results:")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        elif style == "detailed":
            for name, result in results.items():
                if result is not None:
                    self.print_original_style_breakdown(result, name)
        
        elif style == "comprehensive":
            self.print_comparison(results, "summary")
            print("\n" + "="*80)
            print("DETAILED BREAKDOWN:")
            print("="*80)
            self.print_comparison(results, "detailed")
    
    def save_results(self, results: Dict[str, Dict[str, Any]], filename: str) -> None:
        """Save comprehensive benchmark results to JSON with all original detail preserved."""
        import json
        
        def convert_numpy_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_results = {}
        for name, result in results.items():
            if result is None:
                continue
            serializable_results[name] = convert_numpy_types(result)
        
        serializable_results["_metadata"] = {
            "device": str(self.device),
            "num_runs": self.num_runs,
            "use_ptflops": self.use_ptflops,
            "library_version": "0.1.0",
            "comprehensive_analysis": True,
            "original_script_compatibility": True,
            "enhanced_capabilities": True
        }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Comprehensive results with original script depth saved to {filename}")Conv2D
        
        in_channels = input_shape[2]
        
        # Create and use the layer's built-in printing
        if conv_class == KWDSConv2D:
            model = conv_class(in_channels, output_channels, **kwargs)
            model.benchmark_and_print(input_shape, output_channels=output_channels)
        elif conv_class == ODConv2D:
            model = conv_class(in_channels, output_channels, **kwargs)
            model.benchmark_and_print(input_shape, output_channels=output_channels)
        elif conv_class == DeformableConv2D:
            model = conv_class(in_channels, output_channels, **kwargs)
            model.benchmark_and_print(input_shape, output_channels=output_channels)
        else:
            # Use detailed_analysis for other types
            results = self.detailed_analysis(conv_class, input_shape, output_channels, **kwargs)
            print(f"{conv_class.__name__} Analysis:")
            print(f"Parameters: {results.get('total_params', 0) / 1e3:.2f}K")
            print(f"FLOPs: {results.get('total_flops', 0) / 1e6:.2f}M")
            print(f"Mean Latency: {results.get('mean_latency', 0):.2f}ms")
    
    def run_original_style_benchmarks(self, input_shape=(224, 224, 3), output_channels_list=[64, 128, 256]):
        """
        Run benchmarks in the exact style of original scripts.
        This replicates the original main() functions exactly.
        
        Args:
            input_shape: Input shape (H, W, C)
            output_channels_list: List of output channel configurations to test
        """
        print("🔬 RUNNING ORIGINAL-STYLE COMPREHENSIVE BENCHMARKS")
        print("=" * 80)
        
        # Import the original-style benchmark functions
        from ..layers.kernel_warehouse import benchmark_kwds_conv_original_style
        from ..layers.omni_dimensional import benchmark_odconv_original_style
        from ..layers.deformable import benchmark_deformable_convolution_original_style
        from ..layers.traditional import benchmark_traditional_convolution_original_style
        from ..layers.depthwise_separable import benchmark_depthwise_separable_original_style
        from ..layers.dynamic import benchmark_dynamic_convolution_original_style
        
        print("\n1. KERNEL WAREHOUSE DEPTHWISE SEPARABLE:")
        benchmark_kwds_conv_original_style(input_shape, output_channels_list)
        
        print("\n2. OMNI-DIMENSIONAL DYNAMIC CONVOLUTION:")
        benchmark_odconv_original_style(input_shape, output_channels_list)
        
        print("\n3. DEFORMABLE CONVOLUTION v1:")
        benchmark_deformable_convolution_original_style(input_shape, output_channels_list)
        
        print("\n4. TRADITIONAL CONVOLUTION:")
        # Convert to (C, H, W) format for traditional
        trad_input_shape = (input_shape[2], input_shape[0], input_shape[1])
        benchmark_traditional_convolution_original_style(trad_input_shape, output_channels_list)
        
        print("\n5. DEPTHWISE SEPARABLE CONVOLUTION:")
        benchmark_depthwise_separable_original_style(input_shape, output_channels_list)
        
        print("\n6. DYNAMIC CONVOLUTION:")
        dynamic_input_shape = (input_shape[2], input_shape[0], input_shape[1])
        benchmark_dynamic_convolution_original_style(dynamic_input_shape, output_channels_list)
        
        print("\n✅ COMPREHENSIVE ORIGINAL-STYLE BENCHMARKING COMPLETE")
        print("=" * 80)
    
    def advanced_comparison(self, input_shape=(224, 224, 3), output_channels=64, **layer_configs):
        """
        Advanced comparison with sophisticated analysis and custom configurations.
        
        Args:
            input_shape: Input shape (H, W, C)
            output_channels: Number of output channels
            **layer_configs: Custom configurations for each layer type
        """
        print("🔬 ADVANCED SOPHISTICATED COMPARISON")
        print("=" * 60)
        
        from ..layers import KWDSConv2D, ODConv2D, DeformableConv2D, TraditionalConv2D, DepthwiseSeparableConv
        
        # Default configurations
        configs = {
            'kwds': layer_configs.get('kwds', {'kernel_size': 3, 'stride': 2, 'num_kernels': 4}),
            'odconv': layer_configs.get('odconv', {'kernel_size': 3, 'stride': 2, 'num_candidates': 6}),
            'deformable': layer_configs.get('deformable', {'kernel_size': 3, 'stride': 1, 'padding': 1}),
            'traditional': layer_configs.get('traditional', {'kernel_size': 3, 'stride': 2}),
            'depthwise': layer_configs.get('depthwise', {'kernel_size': 3, 'stride': 2})
        }
        
        results = {}
        
        # Analyze each layer with full sophistication
        print("\nDetailed Layer Analysis:")
        print("-" * 40)
        
        for name, (layer_class, config) in [
            ("KWDS", (KWDSConv2D, configs['kwds'])),
            ("ODConv", (ODConv2D, configs['odconv'])),
            ("Deformable", (DeformableConv2D, configs['deformable'])),
            ("Traditional", (TraditionalConv2D, configs['traditional'])),
            ("Depthwise Sep", (DepthwiseSeparableConv, configs['depthwise']))
        ]:
            print(f"\n{name} Analysis:")
            try:
                result = self.detailed_analysis(layer_class, input_shape, output_channels, **config)
                results[name] = result
                
                # Print sophisticated breakdown
                params = result.get('total_params', 0)
                flops = result.get('total_flops', 0)
                latency = result.get('mean_latency', 0)
                
                print(f"  Parameters: {params / 1e3:.2f}K")
                print(f"  FLOPs: {flops / 1e6:.2f}M")
                print(f"  Latency: {latency:.2f}ms (±{result.get('std_latency', 0):.2f})")
                print(f"  P95/P99: {result.get('p95_latency', 0):.2f}/{result.get('p99_latency', 0):.2f}ms")
                
                # Show layer-specific sophisticated details
                if name == "KWDS" and 'dw_mults' in result:
                    print(f"  - Depthwise Mults: {result['dw_mults'] / 1e6:.2f}M")
                    print(f"  - Pointwise Mults: {result['pw_mults'] / 1e6:.2f}M")
                    print(f"  - Dynamic Branch: {result['dynamic_flops'] / 1e6:.2f}M")
                elif name == "ODConv" and 'dw_mults' in result:
                    print(f"  - 4D Attention Analysis:")
                    print(f"    * Depthwise: {result['dw_mults'] / 1e6:.2f}M FLOPs")
                    print(f"    * Pointwise: {result['pw_mults'] / 1e6:.2f}M FLOPs")
                    print(f"    * Dynamic: {result['dynamic_flops'] / 1e6:.2f}M FLOPs")
                    
            except Exception as e:
                print(f"  Error: {e}")
                results[name] = None
        
        return results
    
    def benchmark_single(
        self, 
        conv_class: nn.Module,
        input_shape: Tuple[int, int, int],
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Benchmark a single convolution type with comprehensive analysis.
        """
        # Import utils here to avoid circular imports
        from ..utils import get_comprehensive_analysis
        from ..layers import KWDSConv2D, ODConv2D, DepthwiseSeparableConv, DeformableConv2D
        
        in_channels = input_shape[2]
        if padding is None:
            padding = kernel_size // 2
        
        # Create model based on type
        if conv_class == KWDSConv2D:
            model = conv_class(in_channels, output_channels, kernel_size, stride, **kwargs)
            conv_type = "kernel_warehouse"
        elif conv_class == ODConv2D:
            model = conv_class(in_channels, output_channels, kernel_size, stride, **kwargs)
            conv_type = "omni_dimensional"
        elif conv_class == DepthwiseSeparableConv:
            model = conv_class(in_channels, output_channels, kernel_size, stride, padding)
            conv_type = "depthwise_separable"
        elif conv_class == DeformableConv2D:
            model = conv_class(in_channels, output_channels, kernel_size, stride, padding, **kwargs)
            conv_type = "deformable"
        else:
            model = conv_class(in_channels, output_channels, kernel_size, stride, padding)
            conv_type = "standard"
            
        model = model.to(self.device).eval()
        
        # Create input tensor
        input_tensor = torch.randn(1, in_channels, input_shape[0], input_shape[1]).to(self.device)
        
        # Warm-up
        with torch.no_grad():
            output = model(input_tensor)
        
        # Benchmark latency (exactly like original scripts)
        latencies = []
        for _ in range(self.num_runs):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(input_tensor)
                    
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        latencies = np.array(latencies)
        
        # Get output shape
        output_shape = tuple(output.shape[1:])  # Remove batch dimension
        
        # Get comprehensive analysis (manual + ptflops)
        analysis_input_shape = (in_channels, input_shape[0], input_shape[1])  # ptflops format
        comprehensive_analysis = get_comprehensive_analysis(
            model, analysis_input_shape, conv_type, **kwargs
        )
        
        # Build result with all original script details
        result = {
            "conv_type": conv_type,
            "model_params": sum(p.numel() for p in model.parameters()),
            "output_shape": output_shape,
            "kernel_size": kernel_size,
            "input_shape": input_shape,
            "output_channels": output_channels,
            "stride": stride,
            "padding": padding,
            
            # Latency statistics (exactly like originals)
            "latency_mean": np.mean(latencies),
            "latency_std": np.std(latencies),
            "latency_min": np.min(latencies),
            "latency_max": np.max(latencies),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99),
            
            # Comprehensive analysis
            "comprehensive_analysis": comprehensive_analysis,
        }
        
        return result
    
    def compare_all(
        self,
        input_shape: Tuple[int, int, int],
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare all convolution types with comprehensive analysis.
        """
        results = {}
        
        for name, conv_class in self.conv_types.items():
            try:
                print(f"Benchmarking {name}...")
                result = self.benchmark_single(
                    conv_class, input_shape, output_channels, kernel_size, stride, padding, **kwargs
                )
                results[name] = result
            except Exception as e:
                print(f"Failed to benchmark {name}: {e}")
                results[name] = None
        
        return results
    
    def print_original_style_breakdown(self, result: Dict[str, Any], conv_name: str) -> None:
        """
        Print results in the exact style of your original scripts with full detail breakdown.
        """
        analysis = result["comprehensive_analysis"]
        manual = analysis.get("manual_calculation")
        ptflops = analysis.get("ptflops_analysis")
        
        if conv_name in ["Kernel Warehouse", "ODConv"]:
            # Use the detailed breakdown style from DSKW.py and DSODConv.py
            print(f"\nBenchmarking Results for {conv_name}:")
            print(f"  Output Channels: {result['output_channels']}")
            print(f"  Kernel Size: {result['kernel_size']}")
            print(f"  Stride: {result['stride']}")
            
            if manual:
                print(f"  Output Shape: {manual.get('output_shape', result['output_shape'])}")
                print(f"  Parameters: {manual['total_params'] / 1e3:.2f}K")
                print(f"  Total FLOPs: {manual['total_flops'] / 1e6:.2f}M")
                
                if "depthwise_mults" in manual:
                    print(f"    - Depthwise Mults: {manual['depthwise_mults'] / 1e6:.2f}M")
                    print(f"    - Pointwise Mults: {manual['pointwise_mults'] / 1e6:.2f}M")
                    print(f"    - Dynamic Branch FLOPs: {manual['dynamic_flops'] / 1e6:.2f}M")
                    print(f"    - Convolution Adds: {manual['conv_adds'] / 1e6:.2f}M")
                    print(f"    - Divisions: {manual['divisions'] / 1e6:.2f}M")
        
        elif conv_name == "Deformable v1":
            # Use DCNv1.py style
            print("Deformable Convolution Layer:")
            if ptflops:
                print(f"Total Parameters: {ptflops['params']} ({ptflops['params_str']})")
                print(f"Estimated FLOPs: {ptflops['flops']} (~{ptflops['flops_str']})")
            elif manual:
                print(f"Total Parameters: {manual['total_params']} ({manual['total_params'] / 1e3:.2f}K)")
                print(f"Estimated FLOPs: {manual['total_flops']} (~{manual['total_flops'] / 1e6:.2f}M)")
        
        else:
            # Use ptflops style for others
            print(f"\n{conv_name}:")
            print(f"Kernel Size: {result['kernel_size']}")
            print(f"Input Shape: {result['input_shape']}")
            print(f"Output Channels: {result['output_channels']}")
            
            if ptflops:
                print(f"Parameters: {ptflops['params_str']}")
                print(f"MACs: {ptflops['macs_str']}")
                print(f"FLOPs: {ptflops['flops_str']}")
            elif manual:
                print(f"Parameters: {manual['total_params'] / 1e3:.2f}K")
                print(f"Total FLOPs: {manual['total_flops'] / 1e6:.2f}M")
        
        # Always show latency statistics in original format
        print("Latency Statistics:")
        print(f"  Mean: {result['latency_mean']:.2f}ms")
        print(f"  Std Dev: {result['latency_std']:.2f}ms")
        print(f"  Min: {result['latency_min']:.2f}ms")
        print(f"  Max: {result['latency_max']:.2f}ms")
        print(f"  P95: {result['latency_p95']:.2f}ms")
        print(f"  P99: {result['latency_p99']:.2f}ms")
        print()
    
    def print_comparison(self, results: Dict[str, Dict[str, Any]], style: str = "original") -> None:
        """
        Print comparison results in various styles.
        
        Args:
            results: Results dictionary from compare_all()
            style: "original", "summary", "detailed", or "comprehensive"
        """
        if style == "original":
            # Print each result in original script style
            for name, result in results.items():
                if result is not None:
                    self.print_original_style_breakdown(result, name)
        
        elif style == "summary":
            # Summary table
            headers = [
                "Convolution Type",
                "Parameters", 
                "FLOPs (M)",
                "Mean Latency (ms)",
                "P95 Latency (ms)"
            ]
            
            table_data = []
            for name, result in results.items():
                if result is None:
                    continue
                
                analysis = result["comprehensive_analysis"]
                
                if analysis.get("ptflops_analysis"):
                    ptflops = analysis["ptflops_analysis"]
                    params_str = ptflops["params_str"]
                    flops_str = ptflops["flops_str"]
                elif analysis.get("manual_calculation"):
                    manual = analysis["manual_calculation"]
                    params = manual["total_params"]
                    flops = manual["total_flops"]
                    params_str = f"{params / 1e3:.2f}K"
                    flops_str = f"{flops / 1e6:.1f}M"
                else:
                    params_str = "N/A"
                    flops_str = "N/A"
                    
                row = [
                    name,
                    params_str,
                    flops_str,
                    f"{result['latency_mean']:.2f}",
                    f"{result['latency_p95']:.2f}"
                ]
                table_data.append(row)
            
            print("\nConvolution Comparison Results:")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        elif style == "detailed":
            # Show detailed FLOP breakdown for each
            for name, result in results.items():
                if result is not None:
                    self.print_original_style_breakdown(result, name)
        
        elif style == "comprehensive":
            # Show both summary and detailed
            self.print_comparison(results, "summary")
            print("\n" + "="*80)
            print("DETAILED BREAKDOWN:")
            print("="*80)
            self.print_comparison(results, "detailed")
    
    def save_results(self, results: Dict[str, Dict[str, Any]], filename: str) -> None:
        """
        Save comprehensive benchmark results to JSON with all original detail preserved.
        """
        import json
        
        def convert_numpy_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_results = {}
        for name, result in results.items():
            if result is None:
                continue
            serializable_results[name] = convert_numpy_types(result)
        
        # Add metadata about the benchmark run
        serializable_results["_metadata"] = {
            "device": str(self.device),
            "num_runs": self.num_runs,
            "use_ptflops": self.use_ptflops,
            "library_version": "0.1.0",
            "comprehensive_analysis": True,
            "original_script_compatibility": True,
            "enhanced_capabilities": True
        }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Comprehensive results with original script depth saved to {filename}")
ODConv" and 'dw_mults' in result:
                    print(f"  - 4D Attention Analysis:")
                    print(f"    * Depthwise: {result['dw_mults'] / 1e6:.2f}M FLOPs")
                    print(f"    * Pointwise: {result['pw_mults'] / 1e6:.2f}M FLOPs")
                    print(f"    * Dynamic: {result['dynamic_flops'] / 1e6:.2f}M FLOPs")
                    
            except Exception as e:
                print(f"  Error: {e}")
                results[name] = None
        
        return results
    
    def benchmark_single(
        self, 
        conv_class: nn.Module,
        input_shape: Tuple[int, int, int],
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Benchmark a single convolution type with comprehensive analysis.
        """
        # Import utils here to avoid circular imports
        from ..utils import get_comprehensive_analysis
        from ..layers import KWDSConv2D, ODConv2D, DepthwiseSeparableConv, DeformableConv2D
        
        in_channels = input_shape[2]
        if padding is None:
            padding = kernel_size // 2
        
        # Create model based on type
        if conv_class == KWDSConv2D:
            model = conv_class(in_channels, output_channels, kernel_size, stride, **kwargs)
            conv_type = "kernel_warehouse"
        elif conv_class == ODConv2D:
            model = conv_class(in_channels, output_channels, kernel_size, stride, **kwargs)
            conv_type = "omni_dimensional"
        elif conv_class == DepthwiseSeparableConv:
            model = conv_class(in_channels, output_channels, kernel_size, stride, padding)
            conv_type = "depthwise_separable"
        elif conv_class == DeformableConv2D:
            model = conv_class(in_channels, output_channels, kernel_size, stride, padding, **kwargs)
            conv_type = "deformable"
        else:
            model = conv_class(in_channels, output_channels, kernel_size, stride, padding)
            conv_type = "standard"
            
        model = model.to(self.device).eval()
        
        # Create input tensor
        input_tensor = torch.randn(1, in_channels, input_shape[0], input_shape[1]).to(self.device)
        
        # Warm-up
        with torch.no_grad():
            output = model(input_tensor)
        
        # Benchmark latency
        latencies = []
        for _ in range(self.num_runs):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(input_tensor)
                    
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        latencies = np.array(latencies)
        
        # Get comprehensive analysis
        analysis_input_shape = input_tensor.shape[1:]  # Remove batch dimension
        comprehensive_analysis = get_comprehensive_analysis(
            model, analysis_input_shape, conv_type, **kwargs
        )
        
        # Extract values from analysis
        if comprehensive_analysis.get("manual_calculation"):
            manual = comprehensive_analysis["manual_calculation"]
            total_params = manual.get("total_params", 0)
            total_flops = manual.get("total_flops", 0)
        elif comprehensive_analysis.get("ptflops_analysis"):
            ptflops = comprehensive_analysis["ptflops_analysis"]
            total_params = ptflops.get("params", 0)
            total_flops = ptflops.get("flops", 0)
        else:
            total_params = sum(p.numel() for p in model.parameters())
            total_flops = 0
        
        result = {
            'total_params': total_params,
            'total_flops': total_flops,
            'mean_latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'output_shape': tuple(output.shape[1:]),
            'comprehensive_analysis': comprehensive_analysis
        }
        
        # Add specific metrics from manual calculation if available
        if comprehensive_analysis.get("manual_calculation"):
            manual = comprehensive_analysis["manual_calculation"]
            if "depthwise_mults" in manual:
                result.update({
                    'dw_mults': manual["depthwise_mults"],
                    'pw_mults': manual["pointwise_mults"],
                    'dynamic_flops': manual.get("dynamic_flops", 0)
                })
            elif "pointwise_mults" in manual:
                result.update({
                    'dw_mults': manual.get("depthwise_mults", 0),
                    'pw_mults': manual["pointwise_mults"],
                    'dynamic_flops': manual.get("dynamic_flops", 0)
                })
        
        return result
        
        # Benchmark latency (exactly like original scripts)
        latencies = []
        for _ in range(self.num_runs):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(input_tensor)
                    
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        latencies = np.array(latencies)
        
        # Get output shape
        output_shape = tuple(output.shape[1:])  # Remove batch dimension
        
        # Get comprehensive analysis (manual + ptflops)
        analysis_input_shape = (in_channels, input_shape[0], input_shape[1])  # ptflops format
        comprehensive_analysis = get_comprehensive_analysis(
            model, analysis_input_shape, conv_type, **kwargs
        )
        
        # Build result with all original script details
        result = {
            "conv_type": conv_type,
            "model_params": sum(p.numel() for p in model.parameters()),
            "output_shape": output_shape,
            "kernel_size": kernel_size,
            "input_shape": input_shape,
            "output_channels": output_channels,
            "stride": stride,
            "padding": padding,
            
            # Latency statistics (exactly like originals)
            "latency_mean": np.mean(latencies),
            "latency_std": np.std(latencies),
            "latency_min": np.min(latencies),
            "latency_max": np.max(latencies),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99),
            
            # Comprehensive analysis
            "comprehensive_analysis": comprehensive_analysis,
        }
        
        return result
    
    def compare_all(
        self,
        input_shape: Tuple[int, int, int],
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare all convolution types with comprehensive analysis.
        """
        results = {}
        
        for name, conv_class in self.conv_types.items():
            try:
                print(f"Benchmarking {name}...")
                result = self.benchmark_single(
                    conv_class, input_shape, output_channels, kernel_size, stride, padding, **kwargs
                )
                results[name] = result
            except Exception as e:
                print(f"Failed to benchmark {name}: {e}")
                results[name] = None
        
        return results
    
    def print_original_style_breakdown(self, result: Dict[str, Any], conv_name: str) -> None:
        """
        Print results in the exact style of your original scripts with full detail breakdown.
        """
        analysis = result["comprehensive_analysis"]
        manual = analysis.get("manual_calculation")
        ptflops = analysis.get("ptflops_analysis")
        
        if conv_name in ["Kernel Warehouse", "ODConv"]:
            # Use the detailed breakdown style from DSKW.py and DSODConv.py
            print(f"\nBenchmarking Results for {conv_name}:")
            print(f"  Output Channels: {result['output_channels']}")
            print(f"  Kernel Size: {result['kernel_size']}")
            print(f"  Stride: {result['stride']}")
            
            if manual:
                print(f"  Output Shape: {manual.get('output_shape', result['output_shape'])}")
                print(f"  Parameters: {manual['total_params'] / 1e3:.2f}K")
                print(f"  Total FLOPs: {manual['total_flops'] / 1e6:.2f}M")
                
                if "depthwise_mults" in manual:
                    print(f"    - Depthwise Mults: {manual['depthwise_mults'] / 1e6:.2f}M")
                    print(f"    - Pointwise Mults: {manual['pointwise_mults'] / 1e6:.2f}M")
                    print(f"    - Dynamic Branch FLOPs: {manual['dynamic_flops'] / 1e6:.2f}M")
                    print(f"    - Convolution Adds: {manual['conv_adds'] / 1e6:.2f}M")
                    print(f"    - Divisions: {manual['divisions'] / 1e6:.2f}M")
        
        elif conv_name == "Deformable v1":
            # Use DCNv1.py style
            print("Deformable Convolution Layer:")
            if ptflops:
                print(f"Total Parameters: {ptflops['params']} ({ptflops['params_str']})")
                print(f"Estimated FLOPs: {ptflops['flops']} (~{ptflops['flops_str']})")
            elif manual:
                print(f"Total Parameters: {manual['total_params']} ({manual['total_params'] / 1e3:.2f}K)")
                print(f"Estimated FLOPs: {manual['total_flops']} (~{manual['total_flops'] / 1e6:.2f}M)")
        
        else:
            # Use ptflops style for others
            print(f"\n{conv_name}:")
            print(f"Kernel Size: {result['kernel_size']}")
            print(f"Input Shape: {result['input_shape']}")
            print(f"Output Channels: {result['output_channels']}")
            
            if ptflops:
                print(f"Parameters: {ptflops['params_str']}")
                print(f"MACs: {ptflops['macs_str']}")
                print(f"FLOPs: {ptflops['flops_str']}")
            elif manual:
                print(f"Parameters: {manual['total_params'] / 1e3:.2f}K")
                print(f"Total FLOPs: {manual['total_flops'] / 1e6:.2f}M")
        
        # Always show latency statistics in original format
        print("Latency Statistics:")
        print(f"  Mean: {result['latency_mean']:.2f}ms")
        print(f"  Std Dev: {result['latency_std']:.2f}ms")
        print(f"  Min: {result['latency_min']:.2f}ms")
        print(f"  Max: {result['latency_max']:.2f}ms")
        print(f"  P95: {result['latency_p95']:.2f}ms")
        print(f"  P99: {result['latency_p99']:.2f}ms")
        print()
    
    def print_comparison(self, results: Dict[str, Dict[str, Any]], style: str = "original") -> None:
        """
        Print comparison results in various styles.
        
        Args:
            results: Results dictionary from compare_all()
            style: "original", "summary", "detailed", or "comprehensive"
        """
        if style == "original":
            # Print each result in original script style
            for name, result in results.items():
                if result is not None:
                    self.print_original_style_breakdown(result, name)
        
        elif style == "summary":
            # Summary table
            headers = [
                "Convolution Type",
                "Parameters", 
                "FLOPs (M)",
                "Mean Latency (ms)",
                "P95 Latency (ms)"
            ]
            
            table_data = []
            for name, result in results.items():
                if result is None:
                    continue
                
                analysis = result["comprehensive_analysis"]
                
                if analysis.get("ptflops_analysis"):
                    ptflops = analysis["ptflops_analysis"]
                    params_str = ptflops["params_str"]
                    flops_str = ptflops["flops_str"]
                elif analysis.get("manual_calculation"):
                    manual = analysis["manual_calculation"]
                    params = manual["total_params"]
                    flops = manual["total_flops"]
                    params_str = f"{params / 1e3:.2f}K"
                    flops_str = f"{flops / 1e6:.1f}M"
                else:
                    params_str = "N/A"
                    flops_str = "N/A"
                    
                row = [
                    name,
                    params_str,
                    flops_str,
                    f"{result['latency_mean']:.2f}",
                    f"{result['latency_p95']:.2f}"
                ]
                table_data.append(row)
            
            print("\nConvolution Comparison Results:")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        elif style == "detailed":
            # Show detailed FLOP breakdown for each
            for name, result in results.items():
                if result is not None:
                    self.print_original_style_breakdown(result, name)
        
        elif style == "comprehensive":
            # Show both summary and detailed
            self.print_comparison(results, "summary")
            print("\n" + "="*80)
            print("DETAILED BREAKDOWN:")
            print("="*80)
            self.print_comparison(results, "detailed")
    
    def save_results(self, results: Dict[str, Dict[str, Any]], filename: str) -> None:
        """
        Save comprehensive benchmark results to JSON with all original detail preserved.
        """
        import json
        
        def convert_numpy_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_results = {}
        for name, result in results.items():
            if result is None:
                continue
            serializable_results[name] = convert_numpy_types(result)
        
        # Add metadata about the benchmark run
        serializable_results["_metadata"] = {
            "device": str(self.device),
            "num_runs": self.num_runs,
            "use_ptflops": self.use_ptflops,
            "library_version": "0.1.0",
            "comprehensive_analysis": True,
            "original_script_compatibility": True,
            "enhanced_capabilities": True
        }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Comprehensive results with original script depth saved to {filename}")
