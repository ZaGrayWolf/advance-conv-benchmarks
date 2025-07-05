"""
Exact replications of all original convolution benchmarking scripts.

This module contains standalone, exact replications of each original script
with all the depth, detail, and comprehensiveness preserved.

Available Scripts:
- dskw_exact: EXACT replication of DSKW.py
- dsodconv_exact: EXACT replication of DSODConv.py  
- dcnv1_exact: EXACT replication of DCNv1.py
- ds_ptflops_exact: EXACT replication of DS_Ptflops.py
- d_ptflops_exact: EXACT replication of D_Ptflops.py
- trad_ptf_exact: EXACT replication of trad_ptf.py
- trad_convo_exact: EXACT replication of trad_convo.py

Usage:
    from conv_benchmarks.original_scripts import dskw_exact
    dskw_exact.run_original_dskw_benchmark()
    
    # Or run any script directly:
    python -m conv_benchmarks.original_scripts.dskw_exact
"""

from . import dskw_exact
from . import dsodconv_exact
from . import dcnv1_exact
from . import ds_ptflops_exact
from . import d_ptflops_exact
from . import trad_ptf_exact
from . import trad_convo_exact

# Mapping of original script names to modules
ORIGINAL_SCRIPTS = {
    "DSKW": dskw_exact,
    "DSODConv": dsodconv_exact,
    "DCNv1": dcnv1_exact,
    "DS_Ptflops": ds_ptflops_exact,
    "D_Ptflops": d_ptflops_exact,
    "trad_ptf": trad_ptf_exact,
    "trad_convo": trad_convo_exact,
}

def run_original_script(script_name):
    """
    Run an exact replication of an original script.
    
    Args:
        script_name: Name of the original script ("DSKW", "DSODConv", etc.)
    """
    if script_name not in ORIGINAL_SCRIPTS:
        available = ", ".join(ORIGINAL_SCRIPTS.keys())
        raise ValueError(f"Unknown script '{script_name}'. Available: {available}")
    
    module = ORIGINAL_SCRIPTS[script_name]
    
    # Each module has a run_original_*_benchmark function
    if script_name == "DSKW":
        module.run_original_dskw_benchmark()
    elif script_name == "DSODConv":
        module.run_original_dsodconv_benchmark()
    elif script_name == "DCNv1":
        module.run_original_dcnv1_benchmark()
    elif script_name == "DS_Ptflops":
        module.run_original_ds_ptflops_benchmark()
    elif script_name == "D_Ptflops":
        module.run_original_d_ptflops_benchmark()
    elif script_name == "trad_ptf":
        module.run_original_trad_ptf_benchmark()
    elif script_name == "trad_convo":
        module.run_original_trad_convo_benchmark()

def run_all_original_scripts():
    """
    Run all original scripts in sequence.
    """
    print("🔬 Running ALL Original Scripts in Sequence")
    print("=" * 60)
    
    for script_name in ORIGINAL_SCRIPTS:
        print(f"\n🎯 Running {script_name}:")
        print("-" * 40)
        try:
            run_original_script(script_name)
        except Exception as e:
            print(f"❌ Failed to run {script_name}: {e}")

__all__ = [
    "dskw_exact",
    "dsodconv_exact", 
    "dcnv1_exact",
    "ds_ptflops_exact",
    "d_ptflops_exact",
    "trad_ptf_exact",
    "trad_convo_exact",
    "ORIGINAL_SCRIPTS",
    "run_original_script",
    "run_all_original_scripts",
]
