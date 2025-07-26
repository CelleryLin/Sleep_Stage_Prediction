# Inference Pipeline

This is the inference pipeline for sleep stage prediction using ECG data. It combines local and global classification models to generate accurate per-epoch sleep stage predictions.

## Overview

The Inference Pipeline integrates the outputs from:
1. Local classification models (ConvTran) that process individual ECG rate-level signals
2. Global classification model (ResNet/UNet) that considers the stage-level sequences across the larger scale of the entire sleep session.

This hierarchical approach allows the system to capture both local patterns within specific time windows of ECG rate signals and global patterns across the entire sleep stage sequence.

## Components

### Source Files

- `main.py`: Main entry point for running the inference pipeline
- `pipeline.py`: Implementation of the Pipeline class that orchestrates the inference process

### Pipeline Process

The inference pipeline follows these steps:

1. **Data Loading**: Load ECG rate-level time series data and ground truth sleep stages
2. **Local Classification**: Use multiple specialized ConvTran models to classify each ECG rate segment
3. **Feature Extraction**: Extract stage-level features from local model predictions
4. **Global Classification**: Process the stage-level sequence through a global ResNet/UNet model
5. **Post-processing**: Apply majority voting and other post-processing techniques
6. **Evaluation**: Calculate performance metrics and generate visualizations

## Model Ensemble

The system uses an ensemble of specialized local models, each trained to differentiate specific sleep stages:

1. **REM - NREM classifier** (010000): Distinguishes REM from NREM sleep
2. **Wake/REM/N1 - N2/N3 classifier** (111000): Separates lighter sleep stages from deeper sleep
3. **Wake - N1 classifier** (1n0nnn): Distinguishes wake from N1 sleep
4. **N2 - N3 classifier** (nnn100): Differentiates between N2 and N3 deep sleep stages

The outputs from these local models that analyze ECG rate-level signals are then processed by a global classifier that integrates the stage-level information across the entire sleep session.

## Usage

To run the inference pipeline:

```bash
# From the repository root
python -m Inference_pipeline.src.main
```

## Configuration

The pipeline uses configuration from the `file_paths.py` module to locate:

- Combined data folder containing ECG recordings
- Patient lookup table for train/test splits
- Pre-trained local classification models
- Pre-trained global classification model

## Visualization

The pipeline includes visualization capabilities to display:

1. ECG rate-level time series data
2. Stage-level global prediction heatmaps
3. Ground truth vs. predicted sleep stages

These visualizations help in analyzing the model's performance and understanding the prediction patterns across both ECG rate-level and stage-level sequences.

## Performance Metrics

The pipeline evaluates performance using:

- Matthews Correlation Coefficient (MCC)
- Confusion Matrix
- Additional metrics can be easily added

## Future Improvements

Potential areas for enhancement:

- Hyperparameter tuning for post-processing steps
- Addition of more specialized local classifiers
- Integration of other physiological signals
- Real-time inference capabilities
