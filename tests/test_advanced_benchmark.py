import pytest
import torch
from conv_benchmarks import ConvolutionBenchmark
from conv_benchmarks.layers import KWDSConv2D, ODConv2D

class TestAdvancedBenchmark:
    """Test suite for the enhanced benchmarking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.benchmark = ConvolutionBenchmark(device="cpu", num_runs=5)  # Fast tests
        self.input_shape = (32, 32, 3)
        self.output_channels = 8
    
    def test_detailed_flop_calculation(self):
        """Test detailed FLOP calculation for different convolution types."""
        # Test KWDS convolution
        kwds_result = self.benchmark.benchmark_single(
            KWDSConv2D,
            self.input_shape,
            self.output_channels,
            kernel_size=3,
            stride=1,
            num_kernels=4
        )
        
        assert "flop_details" in kwds_result
        flop_details = kwds_result["flop_details"]
        assert flop_details["type"] == "kernel_warehouse"
        assert "depthwise_mults" in flop_details
        assert "pointwise_mults" in flop_details
        assert "dynamic_flops" in flop_details
        assert flop_details["total_flops"] > 0
    
    def test_odconv_detailed_analysis(self):
        """Test ODConv detailed analysis."""
        odconv_result = self.benchmark.benchmark_single(
            ODConv2D,
            self.input_shape,
            self.output_channels,
            kernel_size=3,
            stride=1,
            num_candidates=4
        )
        
        flop_details = odconv_result["flop_details"]
        assert flop_details["type"] == "omni_dimensional"
        assert "depthwise_mults" in flop_details
        assert "dynamic_flops" in flop_details
        assert flop_details["total_flops"] > 0
    
    def test_comparison_with_detailed_output(self):
        """Test comparing all convolutions with detailed output."""
        results = self.benchmark.compare_all(
            self.input_shape,
            self.output_channels,
            kernel_size=3,
            stride=1
        )
        
        # Should have results for most convolution types
        assert len(results) >= 4
        
        # Check that all successful results have detailed FLOP analysis
        for name, result in results.items():
            if result is not None:
                assert "flop_details" in result
                assert "total_flops" in result["flop_details"]
                assert result["flop_details"]["total_flops"] > 0
    
    def test_ptflops_integration(self):
        """Test ptflops integration when available."""
        from conv_benchmarks.layers import TraditionalConv2D
        
        result = self.benchmark.benchmark_single(
            TraditionalConv2D,
            self.input_shape,
            self.output_channels,
            kernel_size=3,
            stride=1
        )
        
        # Should have either ptflops result or manual calculation
        assert "flop_details" in result
        # ptflops_result might be None if library not available
        if result.get("ptflops_result"):
            assert "flops" in result["ptflops_result"]
            assert "params" in result["ptflops_result"]
    
    def test_save_and_load_results(self):
        """Test saving results to JSON."""
        import tempfile
        import json
        import os
        
        results = self.benchmark.compare_all(
            self.input_shape,
            self.output_channels,
            kernel_size=3,
            stride=1
        )
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_filename = f.name
        
        try:
            self.benchmark.save_results(results, temp_filename)
            
            # Verify file was created and contains valid JSON
            assert os.path.exists(temp_filename)
            
            with open(temp_filename, 'r') as f:
                loaded_data = json.load(f)
            
            # Check that data was serialized correctly
            assert isinstance(loaded_data, dict)
            assert len(loaded_data) > 0
            
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
