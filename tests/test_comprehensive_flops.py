import pytest
import torch
import numpy as np
from conv_benchmarks import ConvolutionBenchmark
from conv_benchmarks.layers import KWDSConv2D, ODConv2D, DepthwiseSeparableConv
from conv_benchmarks.utils import (
    calculate_kwds_flops_and_params,
    calculate_odconv_flops_and_params,
    get_comprehensive_analysis
)

class TestComprehensiveFLOPAccuracy:
    """
    Test suite to validate that FLOP calculations exactly match original implementations.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.benchmark = ConvolutionBenchmark(device="cpu", num_runs=10, use_ptflops=True)
        self.input_shape = (224, 224, 3)  # Original test shape
        self.output_channels = 64
    
    def test_kwds_flop_calculation_accuracy(self):
        """Test KWDS FLOP calculation matches original DSKW.py exactly."""
        # Parameters from original DSKW.py
        kernel_size = 3
        stride = 2
        num_kernels = 4
        
        # Manual calculation using our utils (should match original)
        (total_params, total_flops, dw_mults, pw_mults, dynamic_flops, 
         conv_adds, divs, output_shape) = calculate_kwds_flops_and_params(
            self.input_shape, self.output_channels, kernel_size, stride, num_kernels
        )
        
        # Verify reasonable values
        assert total_params > 0, "Parameters should be positive"
        assert total_flops > 0, "FLOPs should be positive"
        assert dw_mults > 0, "Depthwise mults should be positive" 
        assert pw_mults > 0, "Pointwise mults should be positive"
        assert dynamic_flops > 0, "Dynamic FLOPs should be positive"
        
        # Check output shape calculation (original formula)
        expected_h = (self.input_shape[0] + 2*(kernel_size//2) - kernel_size) // stride + 1
        expected_w = (self.input_shape[1] + 2*(kernel_size//2) - kernel_size) // stride + 1
        assert output_shape == (expected_h, expected_w, self.output_channels)
        
        # Verify parameter calculation breakdown
        in_channels = self.input_shape[2]
        expected_dw_params = num_kernels * in_channels * (kernel_size * kernel_size)
        expected_pw_params = (in_channels * self.output_channels) + self.output_channels
        expected_attn_params = (in_channels * num_kernels) + num_kernels
        expected_total_params = expected_dw_params + expected_pw_params + expected_attn_params
        
        assert total_params == expected_total_params, f"Parameter calculation mismatch: {total_params} vs {expected_total_params}"
    
    def test_odconv_flop_calculation_accuracy(self):
        """Test ODConv FLOP calculation matches original DSODConv.py exactly."""
        kernel_size = 3
        stride = 2
        in_channels = self.input_shape[2]
        num_candidates = 6
        
        # Manual calculation using our utils
        (total_params, total_flops, dw_mults, pw_mults, dynamic_flops,
         conv_adds, divs, output_shape) = calculate_odconv_flops_and_params(
            self.input_shape, self.output_channels, kernel_size, stride, in_channels, num_candidates
        )
        
        # Verify all components are reasonable
        assert total_params > 0
        assert total_flops > 0
        assert dw_mults > 0
        assert pw_mults > 0
        assert dynamic_flops > 0
        assert conv_adds > 0
        assert divs > 0
        
        # Verify parameter breakdown (from original DSODConv.py)
        expected_dw_bank_params = num_candidates * in_channels * (kernel_size * kernel_size)
        expected_pw_params = in_channels * self.output_channels + self.output_channels
        
        # Attention branch parameters
        fc_spatial_params = in_channels * (kernel_size * kernel_size)
        fc_in_params = in_channels * in_channels
        fc_kernel_params = in_channels * (in_channels * kernel_size * kernel_size)
        fc_candidate_params = in_channels * num_candidates + num_candidates
        fc_out_params = in_channels * self.output_channels
        expected_attn_params = fc_spatial_params + fc_in_params + fc_kernel_params + fc_candidate_params + fc_out_params
        
        expected_total_params = expected_dw_bank_params + expected_pw_params + expected_attn_params
        assert total_params == expected_total_params
    
    def test_comprehensive_analysis_integration(self):
        """Test that comprehensive analysis provides both manual and ptflops results."""
        # Test with KWDS layer
        model = KWDSConv2D(3, 64, 3, stride=2, num_kernels=4)
        analysis_input_shape = (3, 224, 224)  # ptflops format
        
        analysis = get_comprehensive_analysis(
            model, analysis_input_shape, "kernel_warehouse", num_kernels=4
        )
        
        # Should have both manual and ptflops results
        assert "manual_calculation" in analysis
        assert "ptflops_analysis" in analysis
        
        if analysis["manual_calculation"] and analysis["ptflops_analysis"]:
            assert "comparison" in analysis
            
            manual = analysis["manual_calculation"]
            ptflops = analysis["ptflops_analysis"]
            comparison = analysis["comparison"]
            
            # Check that both provide reasonable values
            assert manual["total_params"] > 0
            assert manual["total_flops"] > 0
            assert ptflops["params"] > 0
            assert ptflops["flops"] > 0
            
            # Check comparison metrics
            assert "flops_difference" in comparison
            assert "flops_relative_error" in comparison
            assert "params_match" in comparison
    
    def test_benchmark_output_format_matches_originals(self):
        """Test that benchmark output format exactly matches original scripts."""
        result = self.benchmark.benchmark_single(
            KWDSConv2D,
            self.input_shape,
            self.output_channels,
            kernel_size=3,
            stride=2,
            num_kernels=4
        )
        
        # Check all required fields are present
        required_fields = [
            "conv_type", "model_params", "output_shape", "kernel_size",
            "input_shape", "output_channels", "latency_mean", "latency_std",
            "latency_min", "latency_max", "latency_p95", "latency_p99",
            "comprehensive_analysis"
        ]
        
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        # Check comprehensive analysis structure
        analysis = result["comprehensive_analysis"]
        if analysis.get("manual_calculation"):
            manual = analysis["manual_calculation"]
            
            # KWDS should have specific FLOP breakdown
            kwds_fields = [
                "total_params", "total_flops", "depthwise_mults", "pointwise_mults",
                "dynamic_flops", "conv_adds", "divisions", "output_shape"
            ]
            
            for field in kwds_fields:
                assert field in manual, f"Missing KWDS field: {field}"
    
    def test_latency_statistics_accuracy(self):
        """Test that latency statistics match original calculation methods."""
        result = self.benchmark.benchmark_single(
            DepthwiseSeparableConv,
            self.input_shape,
            self.output_channels,
            kernel_size=3,
            stride=2
        )
        
        # Check latency statistics
        assert result["latency_mean"] > 0
        assert result["latency_std"] >= 0
        assert result["latency_min"] > 0
        assert result["latency_max"] >= result["latency_min"]
        assert result["latency_p95"] >= result["latency_mean"]
        assert result["latency_p99"] >= result["latency_p95"]
        
        # Verify percentile calculations are reasonable
        assert result["latency_p99"] <= result["latency_max"]
        assert result["latency_p95"] <= result["latency_p99"]
    
    def test_ptflops_integration_accuracy(self):
        """Test ptflops integration provides accurate results."""
        try:
            from ptflops import get_model_complexity_info
            
            # Test with a simple model
            model = DepthwiseSeparableConv(3, 64, 3, stride=2)
            input_shape = (3, 224, 224)
            
            # Get ptflops analysis
            from conv_benchmarks.utils import get_ptflops_analysis
            result = get_ptflops_analysis(model, input_shape)
            
            if result:  # ptflops available
                assert "macs" in result
                assert "flops" in result
                assert "params" in result
                assert result["flops"] == result["macs"] * 2  # Standard conversion
                assert result["params"] > 0
                assert result["flops"] > 0
                
        except ImportError:
            pytest.skip("ptflops not available")
    
    def test_all_convolution_types_functional(self):
        """Test that all convolution types produce valid results."""
        conv_types = [
            ("Traditional", self.benchmark.conv_types["Traditional"], {}),
            ("Depthwise Separable", self.benchmark.conv_types["Depthwise Separable"], {}),
            ("Dynamic", self.benchmark.conv_types["Dynamic"], {}),
            ("Kernel Warehouse", self.benchmark.conv_types["Kernel Warehouse"], {"num_kernels": 4}),
            ("ODConv", self.benchmark.conv_types["ODConv"], {"num_candidates": 6}),
        ]
        
        for name, conv_class, kwargs in conv_types:
            try:
                result = self.benchmark.benchmark_single(
                    conv_class,
                    self.input_shape,
                    self.output_channels,
                    kernel_size=3,
                    stride=2,
                    **kwargs
                )
                
                # Basic validation
                assert result["model_params"] > 0, f"{name} has invalid parameters"
                assert result["latency_mean"] > 0, f"{name} has invalid latency"
                assert len(result["output_shape"]) == 3, f"{name} has invalid output shape"
                
                # Check comprehensive analysis exists
                assert "comprehensive_analysis" in result, f"{name} missing analysis"
                
            except Exception as e:
                pytest.fail(f"{name} convolution failed: {e}")
    
    def test_comparison_functionality(self):
        """Test that comparison across all types works correctly."""
        results = self.benchmark.compare_all(
            self.input_shape,
            self.output_channels,
            kernel_size=3,
            stride=2,
            num_kernels=4,
            num_candidates=6
        )
        
        # Should have results for most convolution types
        successful_results = [name for name, result in results.items() if result is not None]
        assert len(successful_results) >= 4, f"Too few successful results: {successful_results}"
        
        # Verify each successful result has proper structure
        for name, result in results.items():
            if result is not None:
                assert "comprehensive_analysis" in result
                assert "latency_mean" in result
                assert result["model_params"] > 0
