"""
Complete Inference Pipeline for Sleep Stage Classification

This script provides a complete inference pipeline that:
1. Takes raw ECG data from npz files
2. Runs through local classifiers
3. Runs through global classifier
4. Visualizes results compared to ground truth labels
"""

import os
import sys
import torch

from SleepStageInferencePipeline import SleepStageInferencePipeline

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import file_paths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """Main function to run the inference pipeline"""
    
    # Initialize pipeline
    pipeline = SleepStageInferencePipeline(
        local_model_paths=file_paths.local_model_path_dict,
        global_model_path=file_paths.global_model_path,
        m=5,
        fs=1,
        epoch_len=30
    )
    
    npz_file = os.path.join(file_paths.combined_data_folder, "*_combined.npz")
    output_dir = "./output/inference_results"
    
    if os.path.exists(npz_file):
        results = pipeline.process_file(npz_file, output_dir)
        print("Inference completed successfully!")
    else:
        print(f"File not found: {npz_file}")
        print("Please update the file path in the script.")

if __name__ == "__main__":
    main()
