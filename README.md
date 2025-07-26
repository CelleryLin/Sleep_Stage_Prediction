# Sleep Stage Prediction

This repository contains a comprehensive sleep stage classification system using ECG rate (BPM) signals. The system uses a hierarchical approach with local and global classification models to predict sleep stages from heart rate variability patterns.

## Overview

This project aims to simplify sleep stage detection by using only ECG signals.

The system classifies sleep into five stages:
- Wake
- REM (Rapid Eye Movement)
- N1 (Light Sleep)
- N2 (Intermediate Sleep)
- N3/N4 (Deep Sleep)

Our approach leverages the correlation between heart rate variability and sleep stages, using a hierarchical classification system that combines:
1. **Local Classification**: Analyzes short, windowed **ECG rate-level** signal, with sampling rate of 150, to make initial predictions.
2. **Global Classification**: Considers the broader stage-level sequence, with sampling rate of 1/30, across the entire sleep session to refine predictions.

## Methodology

### Classification Architecture

The classification system consists of two main components:

#### 1. Local Classification

The local classifier analyzes ECG rate-level signals within short time windows to identify patterns associated with specific sleep stages. It employs a ConvTran to process the ECG rate and its first order differential. The local classifier is called "weak classifier", which make only **binary decisions** between key sleep stage pairs. We need at least 4 local classifiers to cover all sleep stages.

Four specialized local classifiers are trained for these binary tasks:
- REM vs. Non-REM
- Wake/REM/N1 vs. N2/N3
- Wake vs. N1
- N2 vs. N3

#### 2. Global Classification

The global classifier analyzes stage-level sequences derived from local classifier outputs. It employs a ResNet or UNet architecture to process the entire sleep session, considering the temporal context and ensuring consistent predictions across the night.


### Inference Pipeline

The hierarchical approach allows the system to:
1. Capture local ECG rate-level patterns within specific time windows
2. Enforce global consistency across the entire sleep stage sequence
3. Leverage specialized binary classifiers for challenging distinctions
4. Integrate predictions through a global model

![Model Architecture]
<!-- Insert model architecture diagram here -->

## Usage

The complete pipeline should be executed in the following order:

1. Data Preprocessing
2. Local Classification Training
3. Global Classification Training
4. Inference Pipeline

For detailed instructions on each component, please refer to the README files in the respective subdirectories:
- [Preprocessing Pipeline](Preprocessing/README.md)
- [Sleep Stage Classifier](Sleep_stage_classifier/README.md)
- [Inference Pipeline](Inference_pipeline/README.md)

## Results

<!-- Insert your results here, including performance metrics, visualizations, etc. -->