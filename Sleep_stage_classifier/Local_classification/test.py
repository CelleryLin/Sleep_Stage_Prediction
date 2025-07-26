"""
Testing script for ConvTran sleep stage classification model.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc

from Dataset.ECGDataset import ECGDataset
from Dataset.load_data import load_data
from Models.model import ConvTran

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import file_paths

def classification_report(y_true, y_pred, mode='macro'):
    """Calculate sensitivity and MCC."""
    cm = confusion_matrix(y_true, y_pred)
    cm = cm / np.sum(cm)
    tn = []; fp = []; fn = []; tp = []
    
    for i in range(cm.shape[0]):
        tp.append(cm[i, i])
        fp.append(np.sum(cm[i, :]) - cm[i, i])
        fn.append(np.sum(cm[:, i]) - cm[i, i])
        tn.append(np.sum(cm) - tp[-1] - fp[-1] - fn[-1])

    tp = np.array(tp); fp = np.array(fp); fn = np.array(fn); tn = np.array(tn)

    # Calculate sensitivity and MCC
    if mode == 'macro':
        sensitivity = np.mean(tp / (tp + fn))
        MCC = np.mean((tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    elif mode == 'micro':
        tp_ = sum(tp); fp_ = sum(fp); fn_ = sum(fn); tn_ = sum(tn)
        sensitivity = tp_ / (tp_ + fn_)
        MCC = (tp_*tn_ - fp_*fn_) / np.sqrt((tp_+fp_)*(tp_+fn_)*(tn_+fp_)*(tn_+fn_))
    else:
        raise ValueError('mode should be macro or micro')
    
    return sensitivity, MCC


def evaluate_model(model, data_loader, loss_module, device, dataset_name="Test"):
    """Evaluate model on given dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (X, time_pos, Y) in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Evaluating {dataset_name}"):
            X = X.to(device)
            time_pos = time_pos.to(device)
            Y = Y.to(device)

            pred = model(X, time_pos)
            loss = loss_module(pred.squeeze(1), Y)
            total_loss += loss.item()

            all_preds.extend(pred.squeeze(1).detach().cpu().numpy())
            all_labels.extend(Y.detach().cpu().numpy())

    total_loss = total_loss / len(data_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return total_loss, all_preds, all_labels


def plot_roc_curve(y_true, y_scores, save_path=None):
    """Plot ROC curve and return best threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Find best threshold (maximize Youden's J statistic)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    return best_threshold, roc_auc


def plot_prediction_distribution(y_true, y_scores, save_path=None):
    """Plot distribution of predictions by true class."""
    ind0 = np.where(y_true == 0)[0]
    ind1 = np.where(y_true == 1)[0]

    plt.figure(figsize=(8, 6))
    plt.boxplot([y_scores[ind0], y_scores[ind1]], labels=['Class 0', 'Class 1'], showfliers=False)
    plt.ylabel('Prediction Score')
    plt.title('Distribution of Prediction Scores by True Class')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def test_model(model_path, config, data_path, threshold=0.5, 
               save_results=True, output_dir=None, use_cache=True):
    """
    Test the trained ConvTran model.
    
    Args:
        model_path: Path to the saved model
        config: Model configuration dictionary
        data_path: Path to the dataset
        threshold: Classification threshold
        save_results: Whether to save results
        output_dir: Directory to save results
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = ConvTran(config, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from: {model_path}")
    
    # Load data
    data_cache_folder = './cache/'

    patient_lut = pd.read_csv(file_paths.patient_lut_file)

    X_train, y_train, X_test, y_test, name_train, name_test = load_data(
        data_path, 
        patient_lut,
        data_cache_folder,
        use_cache=use_cache
    )

    # Create datasets and dataloaders
    train_dataset = ECGDataset(X_train, y_train, transform=False)
    test_dataset = ECGDataset(X_test, y_test, transform=False)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Loss function
    loss_module = torch.nn.BCELoss()
    
    # Evaluate on training set
    print("\n" + "="*50)
    print("TRAINING SET EVALUATION")
    print("="*50)
    train_loss, train_preds, train_labels = evaluate_model(model, train_loader, loss_module, device, "Training")
    train_preds_binary = (train_preds > threshold).astype(int)
    
    train_cm = confusion_matrix(train_labels, train_preds_binary)
    train_sensitivity, train_mcc = classification_report(train_labels, train_preds_binary, mode='macro')
    
    print(f'Training Loss: {train_loss:.4f}')
    print('Confusion Matrix:')
    print(train_cm)
    print(f'Sensitivity: {train_sensitivity:.4f}')
    print(f'MCC: {train_mcc:.4f}')
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    test_loss, test_preds, test_labels = evaluate_model(model, test_loader, loss_module, device, "Test")
    test_preds_binary = (test_preds > threshold).astype(int)
    
    test_cm = confusion_matrix(test_labels, test_preds_binary)
    test_sensitivity, test_mcc = classification_report(test_labels, test_preds_binary, mode='macro')
    
    print(f'Test Loss: {test_loss:.4f}')
    print('Confusion Matrix:')
    print(test_cm)
    print(f'Sensitivity: {test_sensitivity:.4f}')
    print(f'MCC: {test_mcc:.4f}')
    
    # ROC analysis
    print("\n" + "="*50)
    print("ROC ANALYSIS")
    print("="*50)
    _, roc_auc = plot_roc_curve(test_labels, test_preds, 
        save_path=f"{output_dir}/roc_curve.png" if output_dir else None)
    best_threshold, _ = plot_roc_curve(train_labels, train_preds)
    print(f'AUC: {roc_auc:.4f}')
    print(f'Best threshold (Youden\'s J): {best_threshold:.4f}')
    
    # Plot prediction distributions
    plot_prediction_distribution(test_labels, test_preds,
                                save_path=f"{output_dir}/prediction_distribution.png" if output_dir else None)
    
    # Test with best threshold
    if best_threshold != threshold:
        print(f"\nRe-evaluating with best threshold ({best_threshold:.4f}):")
        test_preds_best = (test_preds > best_threshold).astype(int)
        test_cm_best = confusion_matrix(test_labels, test_preds_best)
        test_sensitivity_best, test_mcc_best = classification_report(test_labels, test_preds_best, mode='macro')
        
        print('Confusion Matrix (Best Threshold):')
        print(test_cm_best)
        print(f'Sensitivity: {test_sensitivity_best:.4f}')
        print(f'MCC: {test_mcc_best:.4f}')
    
    # Save results
    if save_results and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'model_path': model_path,
            'threshold': threshold,
            'best_threshold': best_threshold,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_cm': train_cm,
            'test_cm': test_cm,
            'train_sensitivity': train_sensitivity,
            'train_mcc': train_mcc,
            'test_sensitivity': test_sensitivity,
            'test_mcc': test_mcc,
            'roc_auc': roc_auc,
            'test_preds': test_preds,
            'test_labels': test_labels,
            'train_preds': train_preds,
            'train_labels': train_labels
        }
        
        np.savez(f'{output_dir}/test_results.npz', **results)
        print(f"\nResults saved to: {output_dir}/test_results.npz")
    
    return {
        'train_metrics': (train_loss, train_sensitivity, train_mcc),
        'test_metrics': (test_loss, test_sensitivity, test_mcc),
        'roc_auc': roc_auc,
        'best_threshold': best_threshold
    }


if __name__ == "__main__":
    
    model_root = file_paths.local_output_root + '20250726112253_000100/'

    # load training configuration from training_info.npz
    training_info_path = os.path.join(model_root, 'training_info.npz')
    if os.path.exists(training_info_path):
        training_info = np.load(training_info_path, allow_pickle=True)
        config = training_info['config'].item()
        epochs = training_info['epochs'].item()
        batch_size = training_info['batch_size'].item()
        learning_rate = training_info['learning_rate'].item()
        weight_decay = training_info['weight_decay'].item()
    else:
        raise FileNotFoundError(f"Training info not found at {training_info_path}")
    
    # Test the model
    results = test_model(
        model_path=model_root + 'model.pth',
        config=config,
        data_path=file_paths.local_clf_ds_path,
        threshold=0.5,
        save_results=True,
        output_dir=model_root
    )
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    train_loss, train_sens, train_mcc = results['train_metrics']
    test_loss, test_sens, test_mcc = results['test_metrics']
    
    print(f"Training - Loss: {train_loss:.4f}, Sensitivity: {train_sens:.4f}, MCC: {train_mcc:.4f}")
    print(f"Test     - Loss: {test_loss:.4f}, Sensitivity: {test_sens:.4f}, MCC: {test_mcc:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    print(f"Best threshold: {results['best_threshold']:.4f}")
