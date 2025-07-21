import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ECGDataset(Dataset):
    def __init__(self, X, y, transform=False):
        
        self.time_pos = X[:, 0]
        self.X = X[:, 1:]
        self.y = y

        if transform:
            self.transform = get_transforms()
        else:
            self.transform = None
                

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx]).float()
        time_pos = torch.tensor(self.time_pos[idx]).float()
        y = torch.tensor(self.y[idx]).float()

        if self.transform:
            X = self.transform(X)
        
        XX = torch.diff(X, dim=0, prepend=torch.tensor([0.]))
        X = torch.stack((X, XX), dim=0) # (2, 150)

        return X, time_pos, y

class AddNoise(object):
    def __init__(self, noise_factor=0.05, apply_prob=0.5):
        self.noise_factor = noise_factor
        self.apply_prob = apply_prob

    def __call__(self, sample):
        if torch.rand(1) < self.apply_prob:
            sample = sample + self.noise_factor * torch.randn_like(sample)
        return sample
    
class Reverse(object):
    def __init__(self, apply_prob=0.2):
        self.apply_prob = apply_prob
    
    def __call__(self, sample):
        if torch.rand(1) < self.apply_prob:
            return sample.flip(0)
        else:
            return sample
    
class Flip(object):
    def __init__(self, apply_prob=1):
        self.apply_prob = apply_prob
    
    def __call__(self, sample):
        if torch.rand(1) < self.apply_prob:
            mu = sample.mean()
            sample = 2 * mu - sample
        return sample
    
def get_transforms():
    return transforms.Compose([
        AddNoise(apply_prob=0.5),
        # Reverse(apply_prob=0.1),
        Flip(apply_prob=0.1)
    ])