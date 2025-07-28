import os
import sys
import numpy as np
import torch
from torch import nn
from scipy import stats
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Pipeline():
    """
    self.gt: list of ground truth
    self.stage_pred: list of local model prediction
    self.global_pred: list of global model prediction
    self.majority_pred: list of majority vote prediction
    """
    def __init__(self, local_models, global_model, local_test_resultses, all_time_series, all_stages, m=5, fs=1, epoch_len=30, maj_decision_th=0.5):

        self.m = m
        self.fs = fs
        self.epoch_len = epoch_len
        self.maj_decision_th = maj_decision_th

        self.local_models = {k: v.to(device) for k, v in local_models.items()}
        self.global_model = global_model.to(device)
        self.test_resultses = local_test_resultses
        self.all_time_series = all_time_series
        self.all_stages = all_stages

        self.gt = None
        self.stage_pred = None
        self.global_pred = None
        self.majority_pred = None

    def __call__(self):
        self.stage_pred = [self._get_local_result(s)[1] for s in self.all_time_series]
        self.gt = [s[::30] for s in self.all_stages]

        # drop out the len <200
        self.stage_pred = [s for s in self.stage_pred if s.shape[1] >= 200]
        self.gt = [s for s in self.gt if len(s) >= 200]

        self._get_global_result()
        self._get_majority_vote()

    def predict_window(self, time_series, time_pos):
        time_series = torch.tensor(time_series).to(device).float()
        time_pos = torch.tensor(time_pos).to(device).float()
        time_series_diff = torch.diff(time_series, dim=1, prepend=torch.zeros(time_series.shape[0], 1, device=device)).to(device)

        features = torch.cat((time_series.unsqueeze(1), time_series_diff.unsqueeze(1)), dim=1)
        # inference
        pred_proba = []
        pred = []
        for k, model in self.local_models.items():
            output = model(features, time_pos).cpu().detach().numpy()[:,0]
            pred_proba.append(output)
            pred.append((output > self.test_resultses[k]['best_threshold'].item()).astype(int))
        pred = np.array(pred)
        pred_proba = np.array(pred_proba)
        return pred, pred_proba
        
    def _get_local_result(self, time_series):
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


        return pred, pred_proba
    
    def _get_global_result(self):
        # 200 epochs per window, stride = 1
        self.global_pred = []
        if self.stage_pred is None:
            raise ValueError('Please call __call__ method first')
        
        window_len = 200
        stride = 1
        for s in tqdm(self.stage_pred):
            windows = np.array([s[:, i:i+window_len] for i in range(0, s.shape[1]-window_len+1, stride)])
            windows = torch.tensor(windows).to(device).float() # (B, C, 200)
            pred = self.global_model(windows).cpu().detach().numpy() # (B, C, 200)
            # pred = (pred > 0.5).astype(int)

            # stack all windows
            global_pred_per_person = np.ones([pred.shape[0], pred.shape[1], s.shape[1]]) * -1
            for i, idx in enumerate(range(0, s.shape[1]-window_len+1, stride)):
                global_pred_per_person[i, :, idx:idx+window_len] = pred[i]

            self.global_pred.append(global_pred_per_person)

    def _get_majority_vote(self):
        self.majority_pred = []
        if self.global_pred is None:
            raise ValueError('Please call __call__ method first')
        
        for one_global_pred in self.global_pred:
            one_global_pred = one_global_pred.copy() # (B, C, 200)
            one_global_pred[one_global_pred == -1] = np.nan
            one_maj_pred = np.nanmean(one_global_pred, axis=0) # (C, 200)
            one_maj_pred = np.argmax(one_maj_pred, axis=0) # (200,)

            self.majority_pred.append(one_maj_pred)