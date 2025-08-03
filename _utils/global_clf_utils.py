"""
Utility functions for ResNet sleep stage global classification.
Contains shared functions used across training, testing, and deployment scripts.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, matthews_corrcoef

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import file_paths


def prepare_data_splits():
    """Prepare train/test data splits from patient LUT."""
    patient_lut = pd.read_csv(file_paths.patient_lut_file)
    
    train_data_filenames = patient_lut[patient_lut['split'] == 'train']['name'].apply(
        lambda x: x + '_combined.npz'
    ).tolist()
    
    test_data_filenames = patient_lut[patient_lut['split'] == 'test']['name'].apply(
        lambda x: x + '_combined.npz'
    ).tolist()
    
    return train_data_filenames, test_data_filenames

def decoder_4ch(x):
    """
    Decode the 4-channel input to sleep stage classes.
    
    Classification hierarchy:
    1. REM - NREM               (010000)
    2. Wake, REM, N1 - N2, N3   (111000)
    3. Wake - N1                (1n0nnn)
    4. N2 - N3                  (nnn100)
    
    Args:
        x: Input array of shape (batch_size, sequence_length, 4)
    
    Returns:
        Decoded sleep stages: 0=Wake, 1=REM, 2=N1, 3=N2, 4=N3/N4
    """
    x_out = np.zeros((x.shape[0], x.shape[1]))
    
    for b in range(x.shape[0]):
        for i in range(x.shape[1]):
            if x[b, i, 0]:  # Channel 0: REM detection
                x_out[b, i] = 1  # REM
            else:
                if x[b, i, 1]:  # Channel 1: Wake/REM/N1 detection
                    if x[b, i, 2]:  # Channel 2: Wake vs N1
                        x_out[b, i] = 2  # N1
                    else:
                        x_out[b, i] = 0  # Wake
                else:
                    if x[b, i, 3]:  # Channel 3: N2 vs N3
                        x_out[b, i] = 3  # N2
                    else:
                        x_out[b, i] = 4  # N3/N4
                        
    return x_out

def X_decoder(x):
    pass

def plot_confusion_matrix(y_true, y_pred, title, normalize=False, save_path=None):
    """Plot confusion matrix with normalization."""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g' if not normalize else '.2f', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], labels=['Wake', 'REM', 'N1', 'N2', 'N3'])
    plt.yticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], labels=['Wake', 'REM', 'N1', 'N2', 'N3'])
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_predictions(X, Y_true, Y_pred, sample_idx=0, save_path=None):
    """Visualize sample predictions vs ground truth."""
    plt.figure(figsize=(20, 10))
    
    # Plot input features (first 4 channels)
    for i in range(min(4, X.shape[2])):
        plt.subplot(3, 2, i+1)
        plt.plot(X[sample_idx, :, i].cpu().numpy())
        plt.title(f'Input Channel {i+1}')
        plt.grid(True)
    
    # Plot ground truth vs predictions
    plt.subplot(3, 2, 5)
    plt.plot(Y_true[sample_idx], label='Ground Truth', alpha=0.7)
    plt.plot(Y_pred[sample_idx], label='Prediction', alpha=0.7)
    plt.legend()
    plt.title('Ground Truth vs Prediction')
    plt.ylabel('Sleep Stage')
    plt.xlabel('Time (epochs)')
    plt.grid(True)
    
    # Plot difference
    plt.subplot(3, 2, 6)
    diff = Y_true[sample_idx] - Y_pred[sample_idx]
    plt.plot(diff, color='red', alpha=0.7)
    plt.title('Prediction Error')
    plt.ylabel('Error')
    plt.xlabel('Time (epochs)')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_curves(training_info, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        training_info: Dictionary containing training information
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(training_info['train_losses'], label='Train Loss')
    plt.plot(training_info['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_model_and_info(model, training_info, output_dir=None):
    """
    Save model and training information.
    
    Args:
        model: Trained PyTorch model
        training_info: Dictionary containing training information
        output_dir: Directory to save files (optional, uses file_paths if not provided)
    
    Returns:
        tuple: (model_path, info_path)
    """
    import time
    
    if output_dir is None:
        curr = time.strftime("%Y%m%d%H%M%S", time.localtime())
        output_dir = file_paths.global_output_root + curr
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = f'{output_dir}/model.pth'
    torch.save(model.state_dict(), model_path)
    
    # Save training info
    info_path = f'{output_dir}/training_info.npz'
    np.savez(info_path, **training_info)
    
    print(f'Model saved to {model_path}')
    print(f'Training info saved to {info_path}')
    
    return model_path, info_path

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        dict: Dictionary containing various metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return {
        'confusion_matrix': cm,
        'mcc': mcc,
    }

def print_metrics_summary(metrics, prefix=""):
    """
    Print a formatted summary of metrics.
    
    Args:
        metrics: Dictionary containing metrics from calculate_metrics()
        prefix: Optional prefix for the output (e.g., "Training", "Test")
    """
    if prefix:
        print(f"\n{prefix} Metrics:")
        print("-" * (len(prefix) + 9))
    
    cm = metrics['confusion_matrix']
    acc = np.diag(cm).sum() / cm.sum()
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")

class FocalLoss(torch.nn.Module):
    """Focal Loss implementation for handling class imbalance."""
    
    def __init__(self, gamma=1.5, alpha=0.3):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        pt = target * input + (1 - target) * (1 - input)
        ce = -torch.log(pt)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()