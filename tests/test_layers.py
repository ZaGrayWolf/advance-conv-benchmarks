import pytest
import torch
from conv_benchmarks.layers import (
    TraditionalConv2D,
    DepthwiseSeparableConv, 
    DynamicConv2D,
    KWDSConv2D,
    ODConv2D,
)

class TestConvolutionLayers:
    """Test suite for convolution layer implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.input_tensor = torch.randn(1, 3, 32, 32)
        self.in_channels = 3
        self.out_channels = 16
        self.kernel_size = 3
        self.stride = 1
    
    def test_traditional_conv(self):
        """Test TraditionalConv2D layer."""
        layer = TraditionalConv2D(self.in_channels, self.out_channels, self.kernel_size)
        output = layer(self.input_tensor)
        
        assert output.shape[0] == 1  # batch size
        assert output.shape[1] == self.out_channels
        assert output.shape[2] == 32  # same padding
        assert output.shape[3] == 32
    
    def test_depthwise_separable_conv(self):
        """Test DepthwiseSeparableConv layer."""
        layer = DepthwiseSeparableConv(self.in_channels, self.out_channels, self.kernel_size)
        output = layer(self.input_tensor)
        
        assert output.shape[0] == 1
        assert output.shape[1] == self.out_channels
        assert output.shape[2] == 32
        assert output.shape[3] == 32
    
    def test_dynamic_conv(self):
        """Test DynamicConv2D layer."""
        layer = DynamicConv2D(self.in_channels, self.out_channels, self.kernel_size)
        output = layer(self.input_tensor)
        
        assert output.shape[0] == 1
        assert output.shape[1] == self.out_channels
        assert output.shape[2] == 32
        assert output.shape[3] == 32
    
    def test_kwds_conv(self):
        """Test KWDSConv2D layer."""
        layer = KWDSConv2D(self.in_channels, self.out_channels, self.kernel_size, num_kernels=4)
        output = layer(self.input_tensor)
        
        assert output.shape[0] == 1
        assert output.shape[1] == self.out_channels
        assert output.shape[2] == 32
        assert output.shape[3] == 32
    
    def test_odconv(self):
        """Test ODConv2D layer."""
        layer = ODConv2D(self.in_channels, self.out_channels, self.kernel_size, num_candidates=4)
        output = layer(self.input_tensor)
        
        assert output.shape[0] == 1
        assert output.shape[1] == self.out_channels
        assert output.shape[2] == 32
        assert output.shape[3] == 32
    
    def test_different_strides(self):
        """Test layers with different stride values."""
        stride = 2
        layer = TraditionalConv2D(self.in_channels, self.out_channels, self.kernel_size, stride)
        output = layer(self.input_tensor)
        
        # With stride=2, output should be roughly half the input size
        assert output.shape[2] == 16
        assert output.shape[3] == 16
    
    def test_parameter_counts(self):
        """Test that layers have reasonable parameter counts."""
        traditional = TraditionalConv2D(self.in_channels, self.out_channels, self.kernel_size)
        depthwise = DepthwiseSeparableConv(self.in_channels, self.out_channels, self.kernel_size)
        
        traditional_params = sum(p.numel() for p in traditional.parameters())
        depthwise_params = sum(p.numel() for p in depthwise.parameters())
        
        # Depthwise separable should have fewer parameters than traditional
        assert depthwise_params < traditional_params
