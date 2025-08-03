"""
Testing script for ResNet sleep stage global classification model.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, matthews_corrcoef

sys.path.append(os.path.dirname(__file__))
from Dataset.GlobalECGDataset import GlobalECGDataset, custom_collate_fn
from Model.resnet50 import ResNet50, UNet

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from _utils.model_utils import load_global_model
from _utils.global_clf_utils import (
    prepare_data_splits,
    plot_confusion_matrix,
    visualize_predictions,
    calculate_metrics,
    print_metrics_summary
)

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import file_paths
from EnsembledDecoder import build_decoder_tree

decoder = build_decoder_tree(
    encoding_list=list(file_paths.model_path_dict.keys()),
    class_labels=[0,1,2,3,4,4],
)

def evaluate_model(model, data_loader, loss_module, device, n_output_classes, dataset_name="Test"):
    """Evaluate model on given dataset."""
    model.eval()
    total_loss = 0
    
    all_pred_org = []  # Pre-transformer predictions
    all_pred = []      # Post-transformer predictions
    all_Y = []         # Ground truth labels

    with torch.no_grad():
        for i, (X, X_bin, Y, mask) in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Evaluating {dataset_name}"):
            X = X.to(device)
            Y = Y.to(device)
            mask = mask.to(device)

            # One-hot encoding for targets
            Y_onehot = torch.tensor(
                np.eye(n_output_classes)[Y.cpu().numpy().astype(np.int8)]
            ).permute(0, 2, 1).float().to(device)

            # Forward pass
            pred = model(X.permute(0, 2, 1))
            loss = loss_module(pred, Y_onehot)
            total_loss += loss.item()

            # Convert predictions and targets back to class indices
            Y_classes = np.argmax(Y_onehot.cpu().numpy(), axis=1)
            pred_classes = np.argmax(pred.cpu().numpy(), axis=1)

            # Get pre-transformer predictions (from input X)
            X_bin_reshape = X_bin.cpu().numpy().reshape(-1, X_bin.shape[-1])
            pred_org_classes = [decoder(X_bin_reshape[i]) for i in range(X_bin_reshape.shape[0])]

            all_pred_org.extend(pred_org_classes)
            all_pred.extend(pred_classes.reshape(-1))
            all_Y.extend(Y_classes.reshape(-1))

    total_loss = total_loss / len(data_loader)
    
    return (total_loss, 
        np.array(all_pred_org), 
        np.array(all_pred), 
        np.array(all_Y)
    )

def load_training_info(info_path):
    """Load training configuration from .npz file."""
    data = np.load(info_path, allow_pickle=True)
    
    # Convert back to dictionary
    training_info = {}
    for key in data.files:
        item = data[key]
        # Handle scalar strings and other types
        if item.ndim == 0:
            training_info[key] = item.item()
        else:
            training_info[key] = item
    
    return training_info


def test_model(model_dir, training_info_path=None, batch_size=None, save_results=True, output_dir=None):
    """
    Test the trained ResNet/UNet global classification model using training configuration.
    
    Args:
        model_path: Path to the saved model
        batch_size: Batch size for evaluation (if None, uses training batch size)
        save_results: Whether to save results
        output_dir: Directory to save results
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    training_info_path = os.path.join(model_dir, 'training_info.npz')
    
    if not os.path.exists(training_info_path):
        raise FileNotFoundError(f"Training info file not found: {training_info_path}")
    
    print(f"Loading training configuration from: {training_info_path}")
    training_info = load_training_info(training_info_path)
    
    max_len = training_info['max_len']
    n_output_classes = training_info['n_output_classes']
    
    # Use provided batch size or fall back to training batch size
    if batch_size is None:
        batch_size = training_info['batch_size']
    
    # Load model using utility function with correct parameters
    model = load_global_model(model_dir, file_paths.model_path_dict, device)
    
    # Prepare data
    train_data_filenames, test_data_filenames = prepare_data_splits()
    data_base_dir = file_paths.combined_data_folder

    # Create datasets with same configuration as training
    train_dataset = GlobalECGDataset(
        data_base_dir,
        train_data_filenames,
        model_path_dict=file_paths.model_path_dict,
        max_len=max_len
    )

    test_dataset = GlobalECGDataset(
        data_base_dir,
        test_data_filenames,
        model_path_dict=file_paths.model_path_dict,
        max_len=max_len,
        ds='test'
    )

    print(f"Dataset loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )

    # Loss function (use same as training would be ideal, but BCELoss for compatibility)
    loss_module = torch.nn.BCELoss()
    
    # Evaluate on training set
    print("\n" + "="*60)
    print("TRAINING SET EVALUATION")
    print("="*60)
    
    train_loss, train_pred_org, train_pred, train_labels = evaluate_model(
        model, train_loader, loss_module, device, n_output_classes, "Training"
    )
    
    print(f'Training Loss: {train_loss:.4f}')
    print('\nPre-transformer (Input Decoding) Results:')
    train_metrics_org = calculate_metrics(train_labels, train_pred_org)
    print_metrics_summary(train_metrics_org, "Training Pre-transformer")
    plot_confusion_matrix(train_labels, train_pred_org, 
                         'Training Set - Baseline Model',
                         save_path=f"{output_dir}/train_cm_pre.png" if output_dir else None)
    
    print('\nPost-transformer (Model Output) Results:')
    train_metrics_post = calculate_metrics(train_labels, train_pred)
    print_metrics_summary(train_metrics_post, "Training Post-transformer")
    plot_confusion_matrix(train_labels, train_pred, 
                         'Training Set - Global Classifier',
                         save_path=f"{output_dir}/train_cm_post.png" if output_dir else None)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    test_loss, test_pred_org, test_pred, test_labels = evaluate_model(
        model, test_loader, loss_module, device, n_output_classes, "Test"
    )
    
    print(f'Test Loss: {test_loss:.4f}')
    print('\nPre-transformer (Input Decoding) Results:')
    test_metrics_org = calculate_metrics(test_labels, test_pred_org)
    print_metrics_summary(test_metrics_org, "Test Pre-transformer")
    plot_confusion_matrix(test_labels, test_pred_org, 
                         'Test Set - Baseline Model',
                         save_path=f"{output_dir}/test_cm_pre.png" if output_dir else None)
    
    print('\nPost-transformer (Model Output) Results:')
    test_metrics_post = calculate_metrics(test_labels, test_pred)
    print_metrics_summary(test_metrics_post, "Test Post-transformer")
    plot_confusion_matrix(test_labels, test_pred, 
                         'Test Set - Global Classifier',
                         save_path=f"{output_dir}/test_cm_post.png" if output_dir else None)
    
    # Save results
    if save_results and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'model_path': model_dir + '/model.pth',
            'training_info_path': training_info_path,
            'training_config': training_info,
            'max_len': max_len,
            'n_output_classes': n_output_classes,
            'batch_size': batch_size,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_cm_pre': train_metrics_org['confusion_matrix'],
            'train_cm_post': train_metrics_post['confusion_matrix'],
            'test_cm_pre': test_metrics_org['confusion_matrix'],
            'test_cm_post': test_metrics_post['confusion_matrix'],
            'train_mcc_pre': train_metrics_org['mcc'],
            'train_mcc_post': train_metrics_post['mcc'],
            'test_mcc_pre': test_metrics_org['mcc'],
            'test_mcc_post': test_metrics_post['mcc'],
            'train_pred_org': train_pred_org,
            'train_pred': train_pred,
            'train_labels': train_labels,
            'test_pred_org': test_pred_org,
            'test_pred': test_pred,
            'test_labels': test_labels
        }
        
        np.savez(f'{output_dir}/test_results.npz', **results)
        print(f"\nResults saved to: {output_dir}/test_results.npz")
    
    return {
        'train_metrics': (train_loss, train_metrics_org['mcc'], train_metrics_post['mcc']),
        'test_metrics': (test_loss, test_metrics_org['mcc'], test_metrics_post['mcc']),
        'improvement': {
            'train_mcc': train_metrics_post['mcc'] - train_metrics_org['mcc'],
            'test_mcc': test_metrics_post['mcc'] - test_metrics_org['mcc']
        }
    }


if __name__ == "__main__":
    # Testing parameters - now only need model path, everything else loaded from training_info
    model_dir = file_paths.global_output_root + '20250728155424/'
    output_dir = './test_results'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Test the model using configuration from training
    results = test_model(
        model_dir=model_dir,
        batch_size=4096,
        save_results=True,
        output_dir=output_dir
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    train_loss, train_mcc_pre, train_mcc_post = results['train_metrics']
    test_loss, test_mcc_pre, test_mcc_post = results['test_metrics']
    
    print(f"Training   - Loss: {train_loss:.4f}")
    print(f"           - MCC Pre-transformer:  {train_mcc_pre:.4f}")
    print(f"           - MCC Post-transformer: {train_mcc_post:.4f}")
    print(f"           - MCC Improvement:      {results['improvement']['train_mcc']:.4f}")
    print()
    print(f"Test       - Loss: {test_loss:.4f}")
    print(f"           - MCC Pre-transformer:  {test_mcc_pre:.4f}")
    print(f"           - MCC Post-transformer: {test_mcc_post:.4f}")
    print(f"           - MCC Improvement:      {results['improvement']['test_mcc']:.4f}")
