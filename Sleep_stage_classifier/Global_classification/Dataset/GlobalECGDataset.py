import os
import sys
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Local_classification.Models.model import ConvTran

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from stage_code_cvt import stage_code_global

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GlobalECGDataset(Dataset):

    stage_code = stage_code_global

    def __init__(self, data_base_dir, filename_list, model_path_dict, m=5, fs=1, epoch_len=30, max_len=2000, decision_th=0.5):

        self.m = m
        self.fs = fs
        self.epoch_len = epoch_len
        self.max_len = max_len

        all_time_series, all_stages = self.load_data(data_base_dir, filename_list)

        self.models, self.training_infos, self.test_resultses = \
            self.load_model(model_path_dict)
        self.decision_th = decision_th

        epoc = [self.get_epoch(s) for s in all_time_series]
        self.stage_pred = [s[0] for s in epoc]
        self.stage_pred_proba = [s[1] for s in epoc]
        self.gt = [s[::30] for s in all_stages]

        self.stage_pred_proba = [s for s in self.stage_pred_proba if len(s) >= self.max_len]
        self.stage_pred = [s for s in self.stage_pred if len(s) >= self.max_len]
        self.gt = [s for s in self.gt if len(s) >= self.max_len]

        print(len(self.stage_pred))


    def __len__(self):
        return len(self.stage_pred_proba)
    
    def __getitem__(self, idx):
        stage_pred_proba = np.array(self.stage_pred_proba[idx])
        stage_pred = np.array(self.stage_pred[idx])
        gt = np.array(self.gt[idx])

        # padding to max_len
        stage_pred_proba, mask, start = GlobalECGDataset.randcrop(stage_pred_proba, self.max_len)
        stage_pred, _, _ = GlobalECGDataset.randcrop(stage_pred, self.max_len, start=start)
        gt, _, _ = GlobalECGDataset.randcrop(gt, self.max_len, start=start)

        return stage_pred_proba, stage_pred, gt, mask
    

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
    def load_data(data_base_dir, filename_list):
        all_time_series = []
        all_stages = []
        for file in tqdm(filename_list):
            data = np.load(data_base_dir + file, allow_pickle=True)
            data_dict = data['data'].item()

            if data_dict['BPM'] is None or data_dict['STAGE'] is None:
                continue

            time_series = np.array(data_dict['BPM'])[::128] # transform to 1 beat per sec
            stage = np.array(data_dict['STAGE'][:, 0])[::128]

            # encode stage
            stage = np.vectorize(GlobalECGDataset.stage_code.get)(stage.astype(str))

            all_time_series.append(time_series)
            all_stages.append(stage)

        return all_time_series, all_stages
    
    @staticmethod
    def padding(s, max_len):
        """
        Padding 0 to the end of the time series
        s: 1D time series
        """
        s = np.array(s)
        flag = 0
        padded = np.concatenate([s, [flag] * (max_len - len(s))])

        # mask
        mask = np.zeros(max_len)
        mask[:len(s)] = 1

        return padded, mask
    
    @staticmethod
    def randcrop(s, max_len, start=None):
        """
        Randomly crop the time series to max_len
        s: 1D time series
        """
        s = np.array(s)
        if len(s) <= max_len:
            raise ValueError("Time series is shorter than max_len")
        
        if start is None:
            start = np.random.randint(0, len(s) - max_len)

        return s[start:start+max_len], np.ones(max_len), start
    
    @staticmethod
    def load_model(model_path_dict):
        """
        4 models need to be loaded:
        1. REM - NREM (010000)
        2. Wake, REM, N1 - N2, N3 (111000)
        3. Wake - N1 (1n0nnn)
        4. N2 - N3 (nnn100)
        model_path_dict: dict, must contain ['010000', '111000', '1n0nnn', 'nnn100']
        """

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
            models[k].load_state_dict(torch.load(v+'model.pth'))
            models[k].to(device)
            training_infos[k] = training_info
            test_resultses[k] = test_results

        return models, training_infos, test_resultses

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