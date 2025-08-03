import os
import sys
import torch
from torch.utils.data import Dataset
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from _utils.model_utils import load_local_model, load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GlobalECGDataset(Dataset):

    def __init__(self, data_base_dir, filename_list, model_path_dict, m=5, fs=1, epoch_len=30, max_len=2000, ds='train'):

        self.m = m
        self.fs = fs
        self.epoch_len = epoch_len
        self.max_len = max_len

        self.models, self.training_infos, self.test_resultses = \
            load_local_model(model_path_dict)
        
        # Create a cache filename based on the input parameters
        cache_dir = './cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_filename = os.path.join(cache_dir, f"global_ds_{ds}.npz")
        
        # Try to load from cache first
        if os.path.exists(cache_filename):
            print(f"Loading from cache: {cache_filename}")
            cache_data = np.load(cache_filename, allow_pickle=True)
            self.stage_pred = cache_data['stage_pred']
            self.stage_pred_proba = cache_data['stage_pred_proba']
            self.gt = cache_data['gt']
            self.mask = cache_data['mask']
        else:
            print(f"Cache not found. Processing data...")
            all_time_series, all_stages = load_data(data_base_dir, filename_list)
            
            epoc = [self.get_epoch(s) for s in all_time_series]
            self.stage_pred = [s[0] for s in epoc]
            self.stage_pred_proba = [s[1] for s in epoc]
            self.gt = [s[::30] for s in all_stages]
            self.mask = None

            # windowing (if the length is more than max_len.)
            # self.stage_pred = self.windowing(self.stage_pred, self.max_len)
            # self.stage_pred_proba = self.windowing(self.stage_pred_proba, self.max_len)
            # self.gt = self.windowing(self.gt, self.max_len)
            
            # self.stage_pred_proba = [s for s in self.stage_pred_proba if len(s) >= self.max_len]
            # self.stage_pred = [s for s in self.stage_pred if len(s) >= self.max_len]
            # self.gt = [s for s in self.gt if len(s) >= self.max_len]
            # --------------------------------------------------------

            # Padding (if the length is less than max_len.)
            self.stage_pred, _ = self.padding(self.stage_pred, self.max_len)
            self.stage_pred_proba, self.mask = self.padding(self.stage_pred_proba, self.max_len)
            self.gt, _ = self.padding(self.gt, self.max_len)
            # --------------------------------------------------------
            

            # Save to cache
            print(f"Saving to cache: {cache_filename}")
            np.savez(
                cache_filename, 
                stage_pred=self.stage_pred, 
                stage_pred_proba=self.stage_pred_proba, 
                gt=self.gt,
                mask=self.mask
            )


    def __len__(self):
        return len(self.stage_pred_proba)
    
    def __getitem__(self, idx):
        stage_pred_proba = np.array(self.stage_pred_proba[idx])
        stage_pred = np.array(self.stage_pred[idx])
        gt = np.array(self.gt[idx])
        mask = np.array(self.mask[idx]) if self.mask is not None else torch.ones_like(gt)

        return stage_pred_proba, stage_pred, gt, mask
    
    def windowing(self, time_series, max_len):
        """
        Windowing the time series to max_len
        s: list cantains of all time series
        """
        all_window = []
        for s in time_series:
            window_num = len(s) - max_len + 1
            if window_num <= 0:
                continue

            windowed = np.zeros((window_num, max_len))

            # Precompute the indices for the windows
            window_indices = np.arange(window_num)
            window_slices = np.array([np.arange(i, i + max_len) for i in window_indices])

            # Extract the ECG windows using advanced indexing
            windows = s[window_slices]

            all_window.extend(windows)
        
        return all_window

    def get_epoch(self, time_series):
        full_epochs_num = time_series.shape[0] // self.fs // self.epoch_len
        time_pos = np.linspace(0, 1, time_series.shape[0])
        pred = np.zeros(full_epochs_num)
        all_window = []
        all_pos = []

        # padding m//2 epochs at the beginning and end
        time_series = np.concatenate([np.zeros(self.m//2*self.fs*self.epoch_len), time_series, np.zeros(self.m//2*self.fs*self.epoch_len)])
        
        for i in range(full_epochs_num):
            window = time_series[i*self.fs*self.epoch_len:(i+self.m)*self.fs*self.epoch_len]
            pos = time_pos[(i)*self.fs*self.epoch_len]
            # padding
            if window.shape[0] != self.m*self.fs*self.epoch_len:
                window = np.concatenate([np.zeros(self.m*self.fs*self.epoch_len - window.shape[0]), window])
            
            all_window.append(window)
            all_pos.append(pos)

        all_window = np.array(all_window).astype(np.float32)
        all_pos = np.array(all_pos).astype(np.float32)        
        pred, pred_proba = self.predict_window(all_window, all_pos)

        return pred.T, pred_proba.T

    def predict_window(self, time_series, time_pos):
        time_series = torch.tensor(time_series).to(device).float()
        time_pos = torch.tensor(time_pos).to(device).float()
        time_series_diff = torch.diff(time_series, dim=1, prepend=torch.zeros(time_series.shape[0], 1, device=device)).to(device)


        features = torch.cat((time_series.unsqueeze(1), time_series_diff.unsqueeze(1)), dim=1)
        # inference
        pred_proba = []
        pred = []
        for k, model in self.models.items():
            output = model(features, time_pos).cpu().detach().numpy()[:,0]
            pred_proba.append(output)
            pred.append((output > self.test_resultses[k]['best_threshold'].item()).astype(int))
        pred = np.array(pred)
        pred_proba = np.array(pred_proba)
        return pred, pred_proba
    
    @staticmethod
    def padding(time_series, max_len):
        """
        Padding 0 to the end of the time series
        s: 1D time series
        """
        outputs = []
        masks = []
        for s in time_series:
            if len(s) >= max_len:
                outputs.append(s[:max_len])
                masks.append(np.ones(max_len))
                continue

            s = np.array(s)
            if len(s.shape) == 2:
                padded = np.zeros((max_len, s.shape[1]))
                padded[:s.shape[0], :] = s
            else:
                padded = np.zeros(max_len)
                padded[:s.shape[0]] = s

            # mask
            mask = np.zeros(max_len)
            mask[s.shape[0]] = 1

            outputs.append(padded)
            masks.append(mask)

        return np.array(outputs), np.array(masks)

def custom_collate_fn(batch):
    """
    Custom collate function to handle batching of padded sequences
    """
    # Unzip the batch (each item is (stage_pred, gt, mask))
    stage_preds_proba, stage_preds,  gts, masks = zip(*batch)
    
    # Convert to torch tensors
    stage_preds_proba = torch.FloatTensor(np.stack(stage_preds_proba))
    stage_preds = torch.FloatTensor(np.stack(stage_preds))
    gts = torch.FloatTensor(np.stack(gts))
    masks = torch.FloatTensor(np.stack(masks))

    return stage_preds_proba, stage_preds, gts, masks