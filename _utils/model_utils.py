"""
Common utilities for loading data, models, and handling predictions.
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage_code_cvt import stage_code_global


def load_data(data_base_dir, filename_list, stage_code=None):
    """
    Load time series data and stages from combined NPZ files.
    
    Args:
        data_base_dir: Directory containing combined NPZ files
        filename_list: List of filenames to load
        stage_code: Dictionary mapping stage codes to integer labels (defaults to stage_code_global)
        
    Returns:
        tuple: (all_time_series, all_stages)
    """
    if stage_code is None:
        stage_code = stage_code_global
        
    all_time_series = []
    all_stages = []
    
    for file in tqdm(filename_list, desc="Loading data files"):
        data = np.load(os.path.join(data_base_dir, file), allow_pickle=True)
        data_dict = data['data'].item()
        scalar_features = data['scalar_features'].item()

        if data_dict['BPM'] is None or data_dict['STAGE'] is None:
            continue

        if scalar_features['quality'] == 'Unacceptable':
            continue

        time_series = np.array(data_dict['BPM'])[::128]  # transform to 1 beat per sec
        stage = np.array(data_dict['STAGE'][:, 0])[::128]

        # encode stage
        stage = np.vectorize(stage_code.get)(stage.astype(str))

        all_time_series.append(time_series)
        all_stages.append(stage)

    return all_time_series, all_stages


def load_local_model(model_path_dict, device=None):
    """
    Load local classification models.
    
    Args:
        model_path_dict: Dictionary mapping model keys to paths
        device: PyTorch device
        
    Returns:
        tuple: (models, training_infos, test_resultses)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Import here to avoid circular imports
    from Sleep_stage_classifier.Local_classification.Models.model import ConvTran
    
    models = {}
    training_infos = {}
    test_resultses = {}
    
    for k, v in model_path_dict.items():
        if not os.path.exists(v):
            raise ValueError(f"Model path {v} does not exist")
        
        training_info_path = os.path.join(v, 'training_info.npz')
        test_results_path = os.path.join(v, 'test_results.npz')
        
        training_info = np.load(training_info_path, allow_pickle=True)
        test_results = np.load(test_results_path, allow_pickle=True)
        config = training_info['config'].item()

        models[k] = ConvTran(config, num_classes=1).to(device)
        models[k].load_state_dict(torch.load(os.path.join(v, 'model.pth'), map_location=device))
        models[k].eval()  # Set to evaluation mode
        
        training_infos[k] = training_info
        test_resultses[k] = test_results

    return models, training_infos, test_resultses


def load_global_model(model_dir, model_path_dict=None, device=None):
    """
    Load global classification model from checkpoint.
    
    Args:
        model_dir: Directory containing the saved model
        model_path_dict: Dictionary of local model paths (for determining input channels)
        device: PyTorch device
        
    Returns:
        PyTorch model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Import models here to avoid circular imports
    from Sleep_stage_classifier.Global_classification.Model.resnet50 import ResNet50, UNet
    import file_paths
    
    if model_path_dict is None:
        model_path_dict = file_paths.model_path_dict
    
    training_info_path = os.path.join(model_dir, 'training_info.npz')
    
    if not os.path.exists(training_info_path):
        raise FileNotFoundError(f"Training info file not found: {training_info_path}")
    
    print(f"Loading training configuration from: {training_info_path}")
    training_info = np.load(training_info_path, allow_pickle=True)

    model_type = training_info['model_type'].item()
    n_output_classes = training_info['n_output_classes'].item()
    
    # Determine number of input channels from model path dict
    n_input_channels = len(model_path_dict)

    # Initialize model
    if model_type.lower() == 'resnet':
        model = ResNet50(in_channels=n_input_channels, classes=n_output_classes).to(device)
    elif model_type.lower() == 'unet':
        model = UNet(n_channels=n_input_channels, n_classes=n_output_classes, bilinear=True).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load weights
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), map_location=device))
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from: {os.path.join(model_dir, 'model.pth')}")

    return model


def load_training_info(info_path):
    """
    Load training information from a saved NPZ file.
    
    Args:
        info_path: Path to the training_info.npz file
        
    Returns:
        Dictionary containing training configuration and metrics
    """
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Training info file not found: {info_path}")
        
    training_info = dict(np.load(info_path, allow_pickle=True))
    
    # Convert numpy arrays to native Python types for better handling
    processed_info = {}
    for key, value in training_info.items():
        if isinstance(value, np.ndarray) and value.ndim == 0:
            processed_info[key] = value.item()
        else:
            processed_info[key] = value
            
    return processed_info
