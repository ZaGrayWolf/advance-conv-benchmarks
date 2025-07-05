"""
Enhanced utilities for comprehensive FLOP and parameter calculations.
Integrates both manual calculations and ptflops for maximum accuracy.
"""

import math
import torch
import torch.nn as nn

def calculate_conv_flops_and_params(in_channels, out_channels, kernel_size, input_size, stride, padding, dilation=1, groups=1):
    """
    Calculate parameters and FLOPs for standard convolution.
    Based on the original calculate_parameters_and_flops function.
    """
    weight_params = out_channels * (in_channels // groups) * (kernel_size ** 2)
    bias_params = out_channels
    total_params = weight_params + bias_params

    H_in = input_size[0] if isinstance(input_size, (list, tuple)) else input_size
    W_in = input_size[1] if isinstance(input_size, (list, tuple)) else input_size
    
    H_out = math.floor((H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    W_out = math.floor((W_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    flops_per_output = 2 * ((in_channels // groups) * (kernel_size ** 2))
    total_output_elements = H_out * W_out * out_channels
    total_flops = flops_per_output * total_output_elements

    return total_params, total_flops, (H_out, W_out, out_channels)

def calculate_depthwise_separable_flops_and_params(input_shape, out_channels, kernel_size, stride):
    """
    Calculate FLOPs and parameters for depthwise separable convolution.
    Based on the original DS_Ptflops.py calculations.
    """
    in_channels = input_shape[2]
    
    # Output spatial dimensions (assuming padding = kernel_size//2)
    out_height = (input_shape[0] + 2*(kernel_size//2) - kernel_size) // stride + 1
    out_width = (input_shape[1] + 2*(kernel_size//2) - kernel_size) // stride + 1
    
    # Depthwise convolution parameters: in_channels * kernel_size^2 (no bias)
    dw_params = in_channels * (kernel_size * kernel_size)
    
    # Pointwise convolution parameters: in_channels * out_channels (no bias)
    pw_params = in_channels * out_channels
    
    total_params = dw_params + pw_params
    
    # Depthwise convolution FLOPs
    dw_flops = (kernel_size * kernel_size) * out_height * out_width * in_channels
    
    # Pointwise convolution FLOPs
    pw_flops = in_channels * out_channels * out_height * out_width
    
    total_flops = dw_flops + pw_flops
    
    return total_params, total_flops, dw_flops, pw_flops, (out_height, out_width, out_channels)

def calculate_kwds_flops_and_params(input_shape, out_channels, kernel_size, stride, num_kernels=4):
    """
    Calculate FLOPs and parameters for Kernel Warehouse Depthwise Separable convolution.
    Based on the original DSKW.py calculations.
    """
    in_channels = input_shape[2]
    
    # Output spatial dimensions (assuming padding = kernel_size//2)
    out_height = (input_shape[0] + 2*(kernel_size//2) - kernel_size) // stride + 1
    out_width = (input_shape[1] + 2*(kernel_size//2) - kernel_size) // stride + 1

    # Depthwise kernel bank parameters
    dw_params = num_kernels * in_channels * (kernel_size * kernel_size)
    
    # Pointwise conv parameters
    pw_params = (in_channels * out_channels) + out_channels
    
    # Dynamic branch parameters (attention)
    attn_params = (in_channels * num_kernels) + num_kernels

    total_params = dw_params + pw_params + attn_params

    # FLOPs Estimation
    # Depthwise conv FLOPs
    dw_mults = (kernel_size * kernel_size * in_channels) * out_height * out_width
    # Pointwise conv FLOPs
    pw_mults = (in_channels * out_channels) * out_height * out_width
    # Dynamic branch FLOPs
    gap_flops = in_channels * (input_shape[0] * input_shape[1])
    fc_flops = 2 * in_channels * num_kernels
    softmax_flops = 2 * num_kernels
    kernel_elements = in_channels * (kernel_size * kernel_size)
    weighting_flops = (num_kernels + (num_kernels - 1)) * kernel_elements

    dynamic_flops = gap_flops + fc_flops + softmax_flops + weighting_flops

    # Additions for depthwise and pointwise
    dw_adds = ((kernel_size * kernel_size * in_channels) - 1) * out_height * out_width
    pw_adds = ((in_channels * out_channels) - 1) * out_height * out_width
    conv_adds = dw_adds + pw_adds
    
    # Divisions
    divs = out_height * out_width * out_channels

    total_mults = dw_mults + pw_mults + dynamic_flops
    total_flops = total_mults + conv_adds + divs

    output_shape = (out_height, out_width, out_channels)
    return total_params, total_flops, dw_mults, pw_mults, dynamic_flops, conv_adds, divs, output_shape

def calculate_odconv_flops_and_params(input_shape, out_channels, kernel_size, stride, in_channels, num_candidates=6):
    """
    Calculate FLOPs and parameters for ODConv.
    Based on the original DSODConv.py calculations.
    """
    # Candidate kernel bank parameters
    dw_bank_params = num_candidates * in_channels * (kernel_size * kernel_size)
    
    # Pointwise conv parameters
    pw_params = in_channels * out_channels + out_channels
    
    # Attention branch parameters
    fc_spatial_params = in_channels * (kernel_size * kernel_size)
    fc_in_params = in_channels * in_channels
    fc_kernel_params = in_channels * (in_channels * kernel_size * kernel_size)
    fc_candidate_params = in_channels * num_candidates + num_candidates
    fc_out_params = in_channels * out_channels
    attn_params = fc_spatial_params + fc_in_params + fc_kernel_params + fc_candidate_params + fc_out_params
    
    total_params = dw_bank_params + pw_params + attn_params

    # Compute output spatial dimensions
    H_out = (input_shape[0] + 2*(kernel_size//2) - kernel_size) // stride + 1
    W_out = (input_shape[1] + 2*(kernel_size//2) - kernel_size) // stride + 1

    # FLOPs estimation
    dw_mults = (kernel_size * kernel_size * in_channels) * H_out * W_out
    pw_mults = (in_channels * out_channels) * H_out * W_out
    
    # Convolution additions
    dw_adds = ((kernel_size * kernel_size * in_channels) - 1) * H_out * W_out
    pw_adds = ((in_channels * out_channels) - 1) * H_out * W_out
    conv_adds = dw_adds + pw_adds
    
    # Divisions
    divs = H_out * W_out * out_channels

    # Dynamic branch FLOPs
    gap_flops = in_channels * (input_shape[0] * input_shape[1])
    fc_flops = (in_channels * (kernel_size*kernel_size) + in_channels * in_channels + 
                in_channels * num_candidates + in_channels * out_channels)
    softmax_flops = 2 * num_candidates
    kernel_elements = in_channels * (kernel_size * kernel_size)
    weighting_flops = (num_candidates + (num_candidates - 1)) * kernel_elements
    dynamic_flops = gap_flops + fc_flops + softmax_flops + weighting_flops

    total_mults = dw_mults + pw_mults + dynamic_flops
    total_flops = total_mults + conv_adds + divs
    output_shape = (H_out, W_out, out_channels)
    
    return total_params, total_flops, dw_mults, pw_mults, dynamic_flops, conv_adds, divs, output_shape

def get_ptflops_analysis(model, input_shape, print_per_layer=False, verbose=False):
    """
    Use ptflops library for comprehensive FLOP counting with layer-by-layer analysis.
    Exactly replicating the original ptflops usage patterns.
    """
    try:
        from ptflops import get_model_complexity_info
        
        # Get detailed analysis like in the original scripts
        macs, params = get_model_complexity_info(
            model, 
            input_shape, 
            as_strings=False, 
            print_per_layer_stat=print_per_layer,
            verbose=verbose
        )
        
        # Convert MACs to FLOPs (standard conversion)
        flops = macs * 2
        
        return {
            "macs": macs,
            "flops": flops,
            "params": params,
            "macs_str": f"{macs / 1e6:.2f}M",
            "flops_str": f"{flops / 1e6:.2f}M",
            "params_str": f"{params / 1e3:.2f}K" if params < 1e6 else f"{params / 1e6:.2f}M"
        }
        
    except ImportError:
        print("Warning: ptflops not available. Install with: pip install ptflops")
        return None
    except Exception as e:
        print(f"Warning: ptflops analysis failed: {e}")
        return None

def get_comprehensive_analysis(model, input_shape, conv_type=None, **kwargs):
    """
    Get both manual calculations and ptflops analysis for comprehensive comparison.
    This replicates the dual analysis approach from your original scripts.
    """
    result = {
        "manual_calculation": None,
        "ptflops_analysis": None,
        "comparison": None
    }
    
    # Get ptflops analysis
    ptflops_result = get_ptflops_analysis(model, input_shape, print_per_layer=False)
    if ptflops_result:
        result["ptflops_analysis"] = ptflops_result
    
    # Get manual calculation based on convolution type
    if conv_type and hasattr(model, 'in_channels') and hasattr(model, 'out_channels'):
        in_channels = model.in_channels
        out_channels = model.out_channels
        kernel_size = getattr(model, 'kernel_size', 3)
        stride = getattr(model, 'stride', 1)
        
        # Convert input_shape format if needed
        if len(input_shape) == 3 and input_shape[0] == in_channels:
            # (C, H, W) format
            manual_input_shape = (input_shape[1], input_shape[2], input_shape[0])
        else:
            # (H, W, C) format
            manual_input_shape = input_shape
        
        try:
            if conv_type == "depthwise_separable":
                manual_result = calculate_depthwise_separable_flops_and_params(
                    manual_input_shape, out_channels, kernel_size, stride
                )
                result["manual_calculation"] = {
                    "total_params": manual_result[0],
                    "total_flops": manual_result[1], 
                    "depthwise_flops": manual_result[2],
                    "pointwise_flops": manual_result[3],
                    "output_shape": manual_result[4]
                }
            
            elif conv_type == "kernel_warehouse":
                num_kernels = kwargs.get("num_kernels", 4)
                manual_result = calculate_kwds_flops_and_params(
                    manual_input_shape, out_channels, kernel_size, stride, num_kernels
                )
                result["manual_calculation"] = {
                    "total_params": manual_result[0],
                    "total_flops": manual_result[1],
                    "depthwise_mults": manual_result[2],
                    "pointwise_mults": manual_result[3],
                    "dynamic_flops": manual_result[4],
                    "conv_adds": manual_result[5],
                    "divisions": manual_result[6],
                    "output_shape": manual_result[7]
                }
            
            elif conv_type == "omni_dimensional":
                num_candidates = kwargs.get("num_candidates", 6)
                manual_result = calculate_odconv_flops_and_params(
                    manual_input_shape, out_channels, kernel_size, stride, in_channels, num_candidates
                )
                result["manual_calculation"] = {
                    "total_params": manual_result[0],
                    "total_flops": manual_result[1],
                    "depthwise_mults": manual_result[2],
                    "pointwise_mults": manual_result[3],
                    "dynamic_flops": manual_result[4],
                    "conv_adds": manual_result[5],
                    "divisions": manual_result[6],
                    "output_shape": manual_result[7]
                }
            
            else:
                # Standard convolution
                padding = getattr(model, 'padding', kernel_size // 2)
                if isinstance(padding, tuple):
                    padding = padding[0]
                manual_result = calculate_conv_flops_and_params(
                    in_channels, out_channels, kernel_size, 
                    (manual_input_shape[0], manual_input_shape[1]), stride, padding
                )
                result["manual_calculation"] = {
                    "total_params": manual_result[0],
                    "total_flops": manual_result[1],
                    "output_shape": manual_result[2]
                }
                
        except Exception as e:
            print(f"Warning: Manual calculation failed: {e}")
    
    # Compare results if both are available
    if result["manual_calculation"] and result["ptflops_analysis"]:
        manual_flops = result["manual_calculation"]["total_flops"]
        ptflops_flops = result["ptflops_analysis"]["flops"]
        manual_params = result["manual_calculation"]["total_params"]
        ptflops_params = result["ptflops_analysis"]["params"]
        
        result["comparison"] = {
            "flops_difference": abs(manual_flops - ptflops_flops),
            "flops_relative_error": abs(manual_flops - ptflops_flops) / max(manual_flops, ptflops_flops) * 100,
            "params_difference": abs(manual_params - ptflops_params),
            "params_match": manual_params == ptflops_params
        }
    
    return result

# Legacy compatibility functions to match original script interfaces
def estimate_model_flops_ptflops(model, input_shape):
    """
    Legacy function for backward compatibility.
    """
    ptflops_result = get_ptflops_analysis(model, input_shape)
    if ptflops_result:
        return ptflops_result["flops"], ptflops_result["params"]
    else:
        total_params = sum(p.numel() for p in model.parameters())
        return None, total_params
