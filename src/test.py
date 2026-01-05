# src/test_model.py - Test trained model

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Dataset
from simple_model import SimpleBPModel
from simple_loader import create_dataset


# ------------------ Dataset Wrapper (same as train.py) ------------------
class PPGDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        ppg = torch.tensor(item['ppg'], dtype=torch.float32).unsqueeze(0)
        bp = torch.tensor([
            item['systolic'],
            item['diastolic']
        ], dtype=torch.float32)

        return ppg, bp


# ------------------ Load Model ------------------
print("Loading trained model...")
model = SimpleBPModel()
model.load_state_dict(torch.load("models/best_model.pth", map_location="cpu"))
model.eval()

# ------------------ Load Test Data ------------------
print("\nLoading test dataset...")

dataset = create_dataset("data/raw/ppg.csv")

# same split logic as training
np.random.shuffle(dataset)
split = int(len(dataset) * 0.8)
test_dataset = PPGDataset(dataset[split:])

test_loader = DataLoader(test_dataset, batch_size=44, shuffle=False)

print(f"Test samples: {len(test_dataset)}")


# ------------------ Run Predictions ------------------
all_preds = []
all_true = []

with torch.no_grad():
    for signals, bps in test_loader:
        preds = model(signals)
        all_preds.append(preds.numpy())
        all_true.append(bps.numpy())

all_preds = np.vstack(all_preds)
all_true = np.vstack(all_true)


# ------------------ Compute Errors ------------------
sys_error = np.abs(all_preds[:, 0] - all_true[:, 0])
dia_error = np.abs(all_preds[:, 1] - all_true[:, 1])

print("\n" + "="*60)
print("TEST RESULTS")
print("="*60)
print(f"Test samples: {len(all_preds)}")

print("\nSystolic BP:")
print(f"  MAE: {np.mean(sys_error):.2f} mmHg")
print(f"  Max error: {np.max(sys_error):.2f} mmHg")

print("\nDiastolic BP:")
print(f"  MAE: {np.mean(dia_error):.2f} mmHg")
print(f"  Max error: {np.max(dia_error):.2f} mmHg")


# ------------------ Sample Predictions ------------------
print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)

for i in range(5):
    print(f"\nPatient {i+1}:")
    print(f"  True BP:      {all_true[i,0]:.0f}/{all_true[i,1]:.0f} mmHg")
    print(f"  Predicted BP: {all_preds[i,0]:.0f}/{all_preds[i,1]:.0f} mmHg")
    print(f"  Error:        {sys_error[i]:.1f}/{dia_error[i]:.1f} mmHg")


# ------------------ Plot ------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(all_true[:, 0], all_preds[:, 0], alpha=0.6)
axes[0].plot([80, 200], [80, 200], 'r--')
axes[0].set_xlabel("True Systolic (mmHg)")
axes[0].set_ylabel("Predicted Systolic (mmHg)")
axes[0].set_title("Systolic BP Predictions")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(all_true[:, 1], all_preds[:, 1], alpha=0.6, color="coral")
axes[1].plot([40, 120], [40, 120], 'r--')
axes[1].set_xlabel("True Diastolic (mmHg)")
axes[1].set_ylabel("Predicted Diastolic (mmHg)")
axes[1].set_title("Diastolic BP Predictions")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/predictions.png", dpi=150)

print("\nPrediction plot saved to results/predictions.png")
plt.show()
