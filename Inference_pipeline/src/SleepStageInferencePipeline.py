import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, matthews_corrcoef

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from _utils.model_utils import load_local_model, load_global_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SleepStageInferencePipeline:
    """Complete inference pipeline for sleep stage classification"""
    
    def __init__(self, local_model_paths, global_model_path, m=5, fs=1, epoch_len=30, verbose=True):
        """
        Initialize the inference pipeline
        
        Args:
            local_model_paths: Dictionary of local model paths
            global_model_path: Path to global model
            m: Number of epochs for local classification window
            fs: Sampling frequency for BPM (Hz)
            epoch_len: Length of each epoch in seconds
        """
        self.m = m
        self.fs = fs
        self.epoch_len = epoch_len
        self.verbose = verbose
        
        # Load models
        self.local_models, self.local_training_infos, self.local_test_results = load_local_model(local_model_paths)
        self.global_model = load_global_model(global_model_path, local_model_paths, device)
        
        # Stage code mapping
        self.stage_names = {0: 'Wake', 1: 'REM', 2: 'N1', 3: 'N2', 4: 'N3'}
        self.gt_code = {'11': 0, '12': 1, '13': 2, '14': 3, '15': 4, '16': 4 }

    
    def run_local_classification(self, bpm_signal):
        """
        Run local classification on BPM signal windows
        
        Args:
            bpm_signal: Heart rate signal in BPM
            
        Returns:
            local_predictions: Predictions from all local models
            local_probabilities: Probabilities from all local models
        """
        
        epoch_samples = self.fs * self.epoch_len
        full_epochs_num = len(bpm_signal) // epoch_samples
        time_pos = np.linspace(0, 1, len(bpm_signal))
        

        pad_samples = (self.m // 2) * epoch_samples
        padded_signal = np.concatenate([
            np.zeros(pad_samples), 
            bpm_signal, 
            np.zeros(pad_samples)
        ])
        
        # Extract windows for each epoch
        all_windows = []
        all_positions = []
        
        for i in range(full_epochs_num):

            start_idx = i * epoch_samples
            end_idx = (i + self.m) * epoch_samples
            window = padded_signal[start_idx:end_idx]
            pos = time_pos[i * epoch_samples] if i * epoch_samples < len(time_pos) else 1.0
            if len(window) != self.m * epoch_samples:
                window = np.pad(window, (0, self.m * epoch_samples - len(window)), 'constant')
            
            all_windows.append(window)
            all_positions.append(pos)
        
        windows_tensor = torch.tensor(all_windows, dtype=torch.float32).to(device)
        positions_tensor = torch.tensor(all_positions, dtype=torch.float32).to(device)
        windows_diff = torch.diff(
            windows_tensor, dim=1, 
            prepend=torch.zeros(windows_tensor.shape[0], 1, device=device)
        )
        features = torch.stack([windows_tensor, windows_diff], dim=1)
        
        # Run through all local models
        local_predictions = []
        local_probabilities = []
        
        for model_key, model in self.local_models.items():
            model.eval()
            with torch.no_grad():
                output = model(features, positions_tensor).cpu().numpy()[:, 0]
                probabilities = output
                predictions = (output > self.local_test_results[model_key]['best_threshold'].item()).astype(int)
                
                local_predictions.append(predictions)
                local_probabilities.append(probabilities)
        
        return np.array(local_predictions), np.array(local_probabilities)
    
    def run_global_classification(self, local_probabilities, max_len=2000):
        """
        Run global classification on local model outputs
        
        Args:
            local_probabilities: Output probabilities from local models
            max_len: Maximum sequence length for global model
            
        Returns:
            global_predictions: Final sleep stage predictions
            global_probabilities: Final sleep stage probabilities
        """
        
        stage_features = local_probabilities.T # (time_steps, n_models)
        
        # Pad or truncate to max_len
        if len(stage_features) > max_len:
            stage_features = stage_features[:max_len]
            mask = np.ones(max_len)
        else:
            original_len = len(stage_features)
            padded_features = np.zeros((max_len, stage_features.shape[1]))
            padded_features[:original_len] = stage_features
            stage_features = padded_features
            
            mask = np.zeros(max_len)
            mask[:original_len] = 1
        
        features_tensor = torch.tensor(stage_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Run through global model
        self.global_model.eval()
        with torch.no_grad():
            features_reshaped = features_tensor.permute(0, 2, 1)
            output = self.global_model(features_reshaped)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            predictions = np.argmax(probabilities, axis=0)
        
        valid_len = int(np.sum(mask))
        return predictions[:valid_len], probabilities[:, :valid_len]
    
    def visualize_results(self, bpm_signal, ground_truth, predictions, probabilities, 
                         save_path=None, patient_name="Unknown"):
        """
        Visualize the inference results compared to ground truth
        
        Args:
            bpm_signal: Original BPM signal (fs=1)
            ground_truth: Ground truth stage labels (fs=1)
            predictions: Predicted stage labels (fs=1/30)
            probabilities: Prediction probabilities (fs=1/30)
            save_path: Path to save the plot
            patient_name: Name of the patient for the title
        """

        
        epoch_samples = self.fs * self.epoch_len
        bpm_time = np.arange(len(bpm_signal)) / (self.fs * 3600)  # Convert to hours
        pred_time_hours = np.arange(len(predictions)) * self.epoch_len / 3600
        acc, mcc = -1, -1
        if ground_truth is not None:
            # Create ground truth at prediction resolution
            gt_resampled = ground_truth[::epoch_samples][:len(predictions)]
            gt_resampled = np.vectorize(self.gt_code.get)(gt_resampled.astype(str))
            acc, mcc = self.calculate_metrics(gt_resampled, predictions)

            gt_time_hours = pred_time_hours[:len(gt_resampled)]

        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Sleep Stage Classification Results - {patient_name} - ACC: {acc:.2f}, MCC: {mcc:.2f}')
        
        # 1. BPM Signal
        axes[0].plot(bpm_time, bpm_signal, 'b-', alpha=0.7, linewidth=0.5)
        axes[0].set_ylabel('Heart Rate (BPM)')
        axes[0].set_title('Heart Rate Signal')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Ground Truth
        if ground_truth is not None:
            gt_colors = [plt.cm.tab10(stage) for stage in gt_resampled]
            axes[1].scatter(gt_time_hours, gt_resampled, c=gt_colors, s=20, alpha=0.8)
            axes[1].set_ylabel('Sleep Stage')
            axes[1].set_title('Ground Truth Sleep Stages')
            axes[1].set_yticks(list(self.stage_names.keys()))
            axes[1].set_yticklabels(list(self.stage_names.values()))
            axes[1].grid(True, alpha=0.3)
        
        # 3. Predictions
        pred_colors = [plt.cm.tab10(stage) for stage in predictions]
        axes[2].scatter(pred_time_hours, predictions, c=pred_colors, s=20, alpha=0.8)
        axes[2].set_ylabel('Sleep Stage')
        axes[2].set_title('Predicted Sleep Stages')
        axes[2].set_yticks(list(self.stage_names.keys()))
        axes[2].set_yticklabels(list(self.stage_names.values()))
        axes[2].grid(True, alpha=0.3)
        
        # 4. Prediction Probabilities Heatmap
        # im = axes[3].imshow(probabilities, aspect='auto', cmap='viridis', 
        #                    extent=[pred_time_hours[0], pred_time_hours[-1], -0.5, len(self.stage_names)-0.5])
        # axes[3].set_ylabel('Sleep Stage')
        # axes[3].set_xlabel('Time (hours)')
        # axes[3].set_title('Prediction Probabilities')
        # axes[3].set_yticks(list(self.stage_names.keys()))
        # axes[3].set_yticklabels(list(self.stage_names.values()))
        # cbar = plt.colorbar(im, ax=axes[3])
        # cbar.set_label('Probability')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        if self.verbose:
            plt.show()

    def calculate_metrics(self, ground_truth, predictions):
        """
        Calculate performance metrics for the predictions
        
        Args:
            ground_truth: Ground truth stage labels
            predictions: Predicted stage labels
            
        Returns:
            Dictionary with accuracy and MCC
        """
        
        min_len = min(len(ground_truth), len(predictions))
        acc = accuracy_score(ground_truth[:min_len], predictions[:min_len])
        mcc = matthews_corrcoef(ground_truth[:min_len], predictions[:min_len])
        
        return acc, mcc
        
    def process_file(self, npz_file_path, output_dir=None):
        """
        Process a single NPZ file through the complete pipeline
        
        Args:
            npz_file_path: Path to the NPZ file containing ECG and stage data
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing all results
        """
        try:
            data = np.load(npz_file_path, allow_pickle=True)
            data_dict = data['data'].item()

            time_series = np.array(data_dict['BPM'])[::128]
            stage = np.array(data_dict['STAGE'][:, 0])[::128]
            
            # Run local classification
            local_pred, local_prob = self.run_local_classification(time_series)
            
            # Run global classification
            global_pred, global_prob = self.run_global_classification(local_prob)
            
            # Create visualization
            patient_name = Path(npz_file_path).stem.replace('_combined', '')
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f'{patient_name}_results.png')
            else:
                save_path = None

            self.visualize_results(time_series, stage, global_pred, global_prob, 
                                save_path, patient_name)

            results = {
                'patient_name': patient_name,
                'bpm_signal': time_series,
                'ground_truth': stage,
                'local_predictions': local_pred,
                'local_probabilities': local_prob,
                'global_predictions': global_pred,
                'global_probabilities': global_prob
            }
            
            return results
                
        except Exception as e:
            print(f"Error processing file {npz_file_path}: {e}")
            return None
        