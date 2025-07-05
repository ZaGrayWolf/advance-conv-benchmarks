import pytest
import torch
from conv_benchmarks import ConvolutionBenchmark

class TestBenchmark:
    """Test suite for the benchmarking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.benchmark = ConvolutionBenchmark(device="cpu", num_runs=5)  # Fast tests
        self.input_shape = (32, 32, 3)
        self.output_channels = 8
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        assert self.benchmark.device.type == "cpu"
        assert self.benchmark.num_runs == 5
        assert len(self.benchmark.conv_types) == 6
    
    def test_single_benchmark(self):
        """Test benchmarking a single convolution type."""
        from conv_benchmarks.layers import TraditionalConv2D
        
        result = self.benchmark.benchmark_single(
            TraditionalConv2D, 
            self.input_shape, 
            self.output_channels,
            kernel_size=3,
            stride=1
        )
        
        # Check that all expected keys are present
        expected_keys = [
            "parameters", "output_shape", "latency_mean", "latency_std",
            "latency_min", "latency_max", "latency_p95", "latency_p99", "flops_estimate"
        ]
        
        for key in expected_keys:
            assert key in result
        
        # Check reasonable values
        assert result["parameters"] > 0
        assert result["latency_mean"] > 0
        assert result["flops_estimate"] > 0
        assert len(result["output_shape"]) == 3  # (C, H, W)
    
    def test_compare_all(self):
        """Test comparing all convolution types."""
        results = self.benchmark.compare_all(
            self.input_shape,
            self.output_channels,
            kernel_size=3,
            stride=1
        )
        
        # Should have results for most convolution types
        assert len(results) >= 4  # At least 4 should work
        
        # Check that successful results have the right structure
        for name, result in results.items():
            if result is not None:
                assert "parameters" in result
                assert "latency_mean" in result
    
    def test_flops_estimation(self):
        """Test FLOP estimation for different convolution types."""
        from conv_benchmarks.layers import TraditionalConv2D, DepthwiseSeparableConv
        
        traditional_flops = self.benchmark._estimate_flops(
            TraditionalConv2D, self.input_shape, self.output_channels, 3, 1
        )
        
        depthwise_flops = self.benchmark._estimate_flops(
            DepthwiseSeparableConv, self.input_shape, self.output_channels, 3, 1
        )
        
        # Depthwise separable should have fewer FLOPs
        assert depthwise_flops < traditional_flops
        assert traditional_flops > 0
        assert depthwise_flops > 0
