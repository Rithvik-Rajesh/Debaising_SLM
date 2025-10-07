#!/usr/bin/env python3
"""
Example script showing how to use the LoRA benchmarking functions.

Run this from the benchmarking directory or adjust the import path accordingly.
"""

from stereoset import benchmark_with_lora, load_lora_model

def example_usage():
    """
    Example of how to use the LoRA benchmarking functions.
    """
    print("LoRA Benchmarking Example")
    print("=" * 30)
    
    # Option 1: Direct benchmarking (recommended)
    print("\n1. Running full benchmark with LoRA model...")
    reports = benchmark_with_lora(
        lora_model_path="../outputs/models/cda_lora_model",
        output_suffix="_example"
    )
    
    if reports:
        print("✓ Benchmarking completed successfully!")
        
        # Display summary
        print("\nResults Summary:")
        for report in reports:
            bias_type = report['bias_type']
            inter_icat = report['Inter ICAT Score']
            intra_icat = report['Intra ICAT Score']
            print(f"  {bias_type}: Inter={inter_icat:.2f}, Intra={intra_icat:.2f}")
    
    # Option 2: Just load the model (for custom usage)
    print("\n2. Loading LoRA model only...")
    lora_mc, lora_mlm = load_lora_model("../outputs/models/cda_lora_model")
    
    if lora_mc and lora_mlm:
        print("✓ LoRA models loaded successfully!")
        print("  You can now use lora_mc and lora_mlm for custom evaluation")
    
    print("\nExample completed!")

if __name__ == "__main__":
    example_usage()