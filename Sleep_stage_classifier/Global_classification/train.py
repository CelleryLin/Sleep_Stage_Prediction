"""
Training script for ResNet sleep stage global classification model.
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

from Dataset.GlobalECGDataset import GlobalECGDataset, custom_collate_fn
from Model.resnet50 import ResNet50, UNet

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from _utils.global_clf_utils import (
    prepare_data_splits, 
    FocalLoss, 
    plot_training_curves,
    save_model_and_info
)
import file_paths
from stage_code_cvt import stage_code_global


def train_model(model_type='unet', epochs=500, batch_size=128, learning_rate=1e-4, 
                max_len=200, val_split=0.2):
    """
    Train the ResNet/UNet global classification model.
    
    Args:
        model_type: Type of model ('resnet' or 'unet')
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        decision_th: Decision threshold for local models
        max_len: Maximum sequence length
        val_split: Validation split ratio
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    train_data_filenames, test_data_filenames = prepare_data_splits()
    data_base_dir = file_paths.combined_data_folder
    
    print(f"Training files: {len(train_data_filenames)}")
    print(f"Test files: {len(test_data_filenames)}")
    
    # Create datasets
    train_dataset = GlobalECGDataset(
        data_base_dir,
        train_data_filenames,
        model_path_dict=file_paths.model_path_dict,
        max_len=max_len,
        ds='train'
    )

    test_dataset = GlobalECGDataset(
        data_base_dir,
        test_data_filenames,
        model_path_dict=file_paths.model_path_dict,
        max_len=max_len,
        ds='test'
    )

    # Split training data for validation
    train_size = int(len(train_dataset) * (1 - val_split))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0, 
        collate_fn=custom_collate_fn,
    )


    # Initialize model
    if model_type.lower() == 'resnet':  # for binary classification
        n_output_classes = 300
        model = ResNet50(in_channels=len(file_paths.model_path_dict), classes=n_output_classes).to(device)
    elif model_type.lower() == 'unet':
        n_output_classes = len(set(stage_code_global.values()))
        model = UNet(n_channels=len(file_paths.model_path_dict), n_classes=n_output_classes, bilinear=True).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    print(f"Using {model_type.upper()} model")


    # Loss and optimizer
    # loss_module = torch.nn.BCELoss()
    loss_module = FocalLoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)

    # Training loop
    all_train_losses = []
    all_val_losses = []

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for i, (X, _, Y, mask) in tqdm(enumerate(train_loader), total=len(train_loader), 
                                   desc=f"Epoch {epoch+1}/{epochs}"):
            X = X.to(device)
            Y = Y.to(device)
            mask = mask.to(device)

            # One-hot encoding for Y (convert to shape: batch_size, n_classes, sequence_length)
            Y_onehot = torch.tensor(
                np.eye(n_output_classes)[Y.cpu().numpy().astype(np.int8)]
            ).permute(0, 2, 1).float().to(device)

            # Forward pass (reshape X for model input)
            pred = model(X.permute(0, 2, 1))  # Reshape from (B, L, C) to (B, C, L)
            loss = loss_module(pred, Y_onehot)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(train_loader)
        all_train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (X_val, _, Y_val, mask) in enumerate(val_loader):
                X_val = X_val.to(device)
                Y_val = Y_val.to(device)
                mask = mask.to(device)

                # One-hot encoding for validation targets
                Y_val_onehot = torch.tensor(
                    np.eye(n_output_classes)[Y_val.cpu().numpy().astype(np.int8)]
                ).permute(0, 2, 1).float().to(device)

                pred = model(X_val.permute(0, 2, 1))
                loss = loss_module(pred, Y_val_onehot)
                val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            all_val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}')

    # Save model and training info
    training_info = {
        'model_type': model_type,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_len': max_len,
        'n_output_classes': n_output_classes,
        'val_split': val_split,
        'train_losses': all_train_losses,
        'val_losses': all_val_losses,
        'final_train_loss': all_train_losses[-1],
        'final_val_loss': all_val_losses[-1]
    }
    
    model_path, info_path = save_model_and_info(model, training_info)
    
    return model, training_info


if __name__ == "__main__":
    # Training parameters
    model, training_info = train_model(
        model_type='unet',  # 'resnet' or 'unet'
        epochs=50,
        batch_size=1024,
        learning_rate=1e-4,
        max_len=200,
        val_split=0.2
    )

    print(f"Training completed. Model saved in: {file_paths.global_output_root}")
    plot_training_curves(training_info)
