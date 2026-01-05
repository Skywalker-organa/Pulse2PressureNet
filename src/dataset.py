
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class PPGDataset(Dataset):
    def __init__(self, data_path, split='train', train_ratio=0.8):
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        split_idx = int(train_ratio * len(all_data))
        self.data = all_data[:split_idx] if split == 'train' else all_data[split_idx:]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        signal = torch.FloatTensor(sample['ppg']).unsqueeze(0)
        bp = torch.FloatTensor([sample['systolic'], sample['diastolic']])
        return signal, bp

def create_dataloaders(data_path='data/ppg_dataset.pkl', batch_size=32):
    train_dataset = PPGDataset(data_path, split='train')
    test_dataset = PPGDataset(data_path, split='test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader