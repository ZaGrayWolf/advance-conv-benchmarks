"""
EXACT replication of trad_convo.py with all the depth and comprehensiveness.
This module preserves every detail of your original simple traditional convolution implementation.
"""
import torch
import torch.nn as nn
import time

def benchmark_convolution(input_shape, output_channels, kernel_size=(3, 3), stride=(2, 2)):
    """
    EXACT replication of the benchmark_convolution function from trad_convo.py
    """
    model = nn.Conv2d(input_shape[2], output_channels, kernel_size, stride=stride)
    model.eval()
    
    # Generate random input data
    input_data = torch.randn(1, input_shape[2], input_shape[0], input_shape[1])
    
    # Warm-up run
    with torch.no_grad():
        model(input_data)
    
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            model(input_data)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"Input shape: {input_shape}, Output channels: {output_channels}, Avg time: {avg_time:.6f} seconds")

def run_original_trad_convo_benchmark():
    """
    EXACT replication of the original main execution from trad_convo.py
    """
    # Example usage
    input_shape = (224, 224, 3)
    benchmark_convolution(input_shape, 64)   # Small Output
    benchmark_convolution(input_shape, 128)  # Medium Output
    benchmark_convolution(input_shape, 256)  # Large Output

# Make this module directly executable like the original
if __name__ == "__main__":
    run_original_trad_convo_benchmark()
