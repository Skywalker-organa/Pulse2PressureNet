import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader, Dataset
from simple_loader import create_dataset
from advanced_model import CNN_LSTM_Model



# ---------------------------------------------------------
# PyTorch Dataset Wrapper
# ---------------------------------------------------------
class PPGDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        ppg = torch.tensor(item['ppg'], dtype=torch.float32).unsqueeze(0)  # (1,1250)
        bp = torch.tensor([
            item['systolic'],
            item['diastolic']
        ], dtype=torch.float32)

        return ppg, bp


# ---------------------------------------------------------
# Training Helpers
# ---------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for signals, bps in tqdm(loader, desc="Training", leave=False):
        signals = signals.to(device)
        bps = bps.to(device)
        
        predictions = model(signals)
        loss = criterion(predictions, bps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for signals, bps in loader:
            signals = signals.to(device)
            bps = bps.to(device)

            predictions = model(signals)
            loss = criterion(predictions, bps)
            total_loss += loss.item()

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(bps.cpu().numpy())
    
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    mae_systolic = np.mean(np.abs(all_predictions[:, 0] - all_targets[:, 0]))
    mae_diastolic = np.mean(np.abs(all_predictions[:, 1] - all_targets[:, 1]))

    return total_loss / len(loader), mae_systolic, mae_diastolic


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
def plot_results(history, save_path='results/training_curves.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['mae_systolic'], 'g-', linewidth=2)
    axes[1].axhline(5, color='red', linestyle='--', label='Target 5mmHg')
    axes[1].axhline(8, color='orange', linestyle='--', label='Acceptable 8mmHg')
    axes[1].set_title("Systolic MAE")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history['mae_diastolic'], 'm-', linewidth=2)
    axes[2].axhline(5, color='red', linestyle='--', label='Target 5mmHg')
    axes[2].axhline(8, color='orange', linestyle='--', label='Acceptable 8mmHg')
    axes[2].set_title("Diastolic MAE")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")


# ---------------------------------------------------------
# MAIN TRAINING FUNCTION
# ---------------------------------------------------------
def train_model(epochs=30, batch_size=32, learning_rate=0.001):
    print("="*70)
    print("TRAINING BLOOD PRESSURE PREDICTION MODEL")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ------------------ Load Data ------------------
    print("\n1. Loading data...")
    dataset = create_dataset("data/raw/ppg.csv")

    np.random.shuffle(dataset)
    split = int(len(dataset) * 0.8)

    train_dataset = PPGDataset(dataset[:split])
    test_dataset = PPGDataset(dataset[split:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    # ------------------ Model ------------------
    print("\n2. Creating model...")
    model = CNN_LSTM_Model().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------ Training Setup ------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {'train_loss': [], 'val_loss': [], 'mae_systolic': [], 'mae_diastolic': []}
    best_mae = float('inf')

    print("\n4. Training...")
    print("-"*70)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, mae_sys, mae_dia = evaluate(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mae_systolic'].append(mae_sys)
        history['mae_diastolic'].append(mae_dia)

        avg_mae = (mae_sys + mae_dia) / 2

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train: {train_loss:.2f} | Val: {val_loss:.2f} | "
              f"Sys MAE: {mae_sys:.2f} | Dia MAE: {mae_dia:.2f} | Avg: {avg_mae:.2f}")

        if avg_mae < best_mae:
            best_mae = avg_mae
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"  ✓ Best model saved (MAE {best_mae:.2f})")

    print("-"*70)
    print(f"Training complete! Best MAE = {best_mae:.2f} mmHg")

    plot_results(history)
    return model, history


if __name__ == "__main__":
    train_model(epochs=30, batch_size=32, learning_rate=0.001)
    print("\nDone! Check results/ folder for plots.")
