"""
Conv Benchmarks: A comprehensive benchmarking suite for advanced convolution operations.

This library EXACTLY replicates and enhances your original convolution benchmarking scripts,
providing both exact original functionality and enhanced unified capabilities.
"""

__version__ = "0.1.0"
__author__ = "Abhudaya Singh"
__email__ = "your.email@example.com"

# Import core components directly to avoid circular imports
from .benchmarks.benchmark import ConvolutionBenchmark

from .layers import (
    TraditionalConv2D,
    DepthwiseSeparableConv,
    DeformableConv2D,
    DynamicConv2D,
    KWDSConv2D,
    ODConv2D,
)

from .utils import (
    calculate_conv_flops_and_params,
    calculate_depthwise_separable_flops_and_params,
    calculate_kwds_flops_and_params,
    calculate_odconv_flops_and_params,
    get_comprehensive_analysis,
    get_ptflops_analysis,
)

# Import original scripts module
from . import original_scripts

# Convenience functions for original script replication
def replicate_dskw(input_shape=(224, 224, 3)):
    """Quick access to DSKW.py replication."""
    from .original_scripts import run_original_script
    run_original_script("DSKW")

def replicate_dsodconv(input_shape=(224, 224, 3)):
    """Quick access to DSODConv.py replication."""
    from .original_scripts import run_original_script
    run_original_script("DSODConv")

def replicate_dcnv1(input_shape=(224, 224, 3)):
    """Quick access to DCNv1.py replication."""
    from .original_scripts import run_original_script
    run_original_script("DCNv1")

def replicate_all_originals():
    """Quick access to run all original scripts."""
    from .original_scripts import run_all_original_scripts
    run_all_original_scripts()

__all__ = [
    # Main benchmarking class
    "ConvolutionBenchmark",
    
    # Quick access functions to original scripts
    "replicate_dskw",
    "replicate_dsodconv",
    "replicate_dcnv1", 
    "replicate_all_originals",
    
    # Convolution layer implementations
    "TraditionalConv2D",
    "DepthwiseSeparableConv", 
    "DeformableConv2D",
    "DynamicConv2D",
    "KWDSConv2D",
    "ODConv2D",
    
    # FLOP calculation utilities (exact from originals)
    "calculate_conv_flops_and_params",
    "calculate_depthwise_separable_flops_and_params",
    "calculate_kwds_flops_and_params", 
    "calculate_odconv_flops_and_params",
    
    # Comprehensive analysis utilities
    "get_comprehensive_analysis",
    "get_ptflops_analysis",
    
    # Original scripts module for direct access
    "original_scripts",
]

# Library metadata
__title__ = "conv-benchmarks"
__description__ = "Comprehensive benchmarking suite with exact original script replication"
__url__ = "https://github.com/yourusername/conv-benchmarks"
__license__ = "MIT"
__copyright__ = "Copyright 2025 Abhudaya Singh"

def get_version():
    """Get the current version of the library."""
    return __version__

def check_dependencies():
    """Check if optional dependencies are available."""
    dependencies = {}
    
    try:
        import torch
        dependencies["torch"] = torch.__version__
    except ImportError:
        dependencies["torch"] = None
    
    try:
        import torchvision
        dependencies["torchvision"] = torchvision.__version__
    except ImportError:
        dependencies["torchvision"] = None
    
    try:
        import ptflops
        dependencies["ptflops"] = True
    except ImportError:
        dependencies["ptflops"] = False
    
    try:
        import tabulate
        dependencies["tabulate"] = True
    except ImportError:
        dependencies["tabulate"] = False
    
    return dependencies

# Quick start example
def quick_benchmark(input_shape=(224, 224, 3), output_channels=64, **kwargs):
    """Quick benchmark of all convolution types."""
    benchmark = ConvolutionBenchmark(num_runs=50)
    return benchmark.compare_all(input_shape, output_channels, **kwargs)
