#!/usr/bin/env python3
"""
Script to run bias benchmarking with the trained LoRA model.

This script loads the LoRA model and performs bias evaluation using StereoSet dataset.
"""

import sys
import os

# Add the benchmarking directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'benchmarking'))

from stereoset import benchmark_with_lora

def main():
    """
    Main function to run LoRA bias benchmarking.
    """
    print("🚀 Starting LoRA Bias Benchmarking")
    print("="*50)
    
    # Path to the LoRA model (relative to script location)
    lora_model_path = "outputs/models/cda_lora_model"
    
    # Run benchmarking
    reports = benchmark_with_lora(
        lora_model_path=lora_model_path,
        output_suffix="_lora"
    )
    
    if reports:
        print("\n🎉 Benchmarking Results Summary:")
        print("-" * 40)
        
        for report in reports:
            bias_type = report.get('bias_type', 'Unknown')
            inter_icat = report.get('Inter ICAT Score', 0)
            intra_icat = report.get('Intra ICAT Score', 0)
            
            print(f"{bias_type.capitalize()} Bias:")
            print(f"  Inter ICAT: {inter_icat:.2f}")
            print(f"  Intra ICAT: {intra_icat:.2f}")
        
        print("\n📊 Detailed reports saved as:")
        print("  • bias_performance_lora_report.txt")
        print("  • bias_performance_lora_report.json")
        
    else:
        print("\n❌ Benchmarking failed. Please check the LoRA model path and dependencies.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)