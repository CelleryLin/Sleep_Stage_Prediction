"""
Training script for ConvTran sleep stage classification model.
"""

import os
import sys
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from Dataset.ECGDataset import ECGDataset
from Dataset.load_data import load_data
from Models.model import ConvTran
from Models.optimizers import RAdam

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from stage_code_cvt import stage_code
import file_paths


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""
    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


def train_model(config, data_path, epochs=50, batch_size=512, 
                learning_rate=5e-3, weight_decay=1e-5, use_cache=True):
    """
    Train the ConvTran model.
    
    Args:
        config: Model configuration dictionary
        data_path: Path to the dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        use_cache: Whether to use cached data
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_cache_folder = './cache/'
    convtran_output_root = file_paths.convtran_output_root
    

    patient_lut = pd.read_csv(file_paths.patient_lut_file)

    X_train, y_train, X_test, y_test, name_train, name_test = load_data(
        data_path, 
        patient_lut,
        data_cache_folder,
        use_cache=use_cache
    )

    print(f"Training data: {len(X_train)} samples")
    print(f"Test data: {len(X_test)} samples")
    print(f"Class distribution - 1s: {np.sum(y_train == 1)}, 0s: {np.sum(y_train == 0)}")

    # Create datasets and dataloaders
    train_dataset = ECGDataset(X_train, y_train, transform=False)
    test_dataset = ECGDataset(X_test, y_test, transform=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = ConvTran(config, num_classes=1).to(device)
    
    # Loss and optimizer
    loss_module = torch.nn.BCELoss()
    optimizer = RAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    all_train_losses = []
    all_val_losses = []

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for i, (X, time_pos, Y) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}"):
            X = X.to(device)
            time_pos = time_pos.to(device)
            Y = Y.to(device)

            pred = model(X, time_pos)
            loss = loss_module(pred.squeeze(1), Y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

            with torch.no_grad():
                epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(train_loader)
        all_train_losses.append(epoch_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, (X, time_pos, Y) in enumerate(test_loader):
                X = X.to(device)
                time_pos = time_pos.to(device)
                Y = Y.to(device)

                pred = model(X, time_pos)
                loss = loss_module(pred.squeeze(1), Y)
                val_loss += loss.item()

            val_loss = val_loss / len(test_loader)
            all_val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}')

    # Save model
    curr = time.strftime("%Y%m%d%H%M%S", time.localtime())
    output_dir = convtran_output_root + curr
    os.makedirs(output_dir, exist_ok=True)
    model_path = f'{output_dir}/model.pth'
    torch.save(model.state_dict(), model_path)
    
    
    # Save training configuration
    training_info = {
        'config': config,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'train_losses': all_train_losses,
        'val_losses': all_val_losses,
        'final_train_loss': all_train_losses[-1],
        'final_val_loss': all_val_losses[-1]
    }
    
    np.savez(f'{output_dir}/training_info.npz', **training_info)
    print(f'Model saved to {model_path}')
    print(f'Training info saved to {output_dir}/training_info.npz')
    
    return model, training_info

def plot_training_curves(training_info):
    """
    Plot training and validation loss curves.
    
    Args:
        training_info: Dictionary containing training information
    """
    plt.figure(figsize=(10, 6))
    plt.plot(training_info['train_losses'], label='Train Loss')
    plt.plot(training_info['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Model configuration
    config = {
        'max_len': 150,
        'channel_size': 2,
        'seq_len': 150,
        'emb_size': 16,
        'num_heads': 8,
        'dim_ff': 256,
        'Fix_pos_encode': 'tAPE',
        'Rel_pos_encode': 'eRPE',
        'dropout': 0.01,
    }
    
    # Training parameters
    data_path = 'F:/Cellery/merry/data/label_window/ECG_Rate_010000_pos/'

    
    # Train the model
    model, training_info = train_model(
        config=config,
        data_path=data_path,
        epochs=20,
        batch_size=1024,
        learning_rate=5e-3,
        weight_decay=1e-5,
        use_cache=True
    )
    
    print(f"Training completed. Model saved in: {file_paths.convtran_output_root}")
    plot_training_curves(training_info)
