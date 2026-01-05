# src/evaluate.py - Advanced Model Evaluation

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

from advanced_model import CNN_LSTM_Model
from dataset import create_dataloaders

# Set style
sns.set_style("whitegrid")

# Load model
print("Loading model...")
model = CNN_LSTM_Model()
model.load_state_dict(torch.load('models/best_model.pth', map_location='cpu'))
model.eval()

# Load test data
_, test_loader = create_dataloaders('data/ppg_dataset.pkl', batch_size=1000)

# Get all predictions
all_preds = []
all_true = []

with torch.no_grad():
    for signals, bps in test_loader:
        preds = model(signals)
        all_preds.append(preds.numpy())
        all_true.append(bps.numpy())

all_preds = np.vstack(all_preds)
all_true = np.vstack(all_true)

# Calculate metrics
sys_error = all_preds[:, 0] - all_true[:, 0]
dia_error = all_preds[:, 1] - all_true[:, 1]

mae_sys = np.mean(np.abs(sys_error))
mae_dia = np.mean(np.abs(dia_error))

rmse_sys = np.sqrt(np.mean(sys_error ** 2))
rmse_dia = np.sqrt(np.mean(dia_error ** 2))

# Correlation
corr_sys, p_sys = stats.pearsonr(all_true[:, 0], all_preds[:, 0])
corr_dia, p_dia = stats.pearsonr(all_true[:, 1], all_preds[:, 1])

# Print report
print("\n" + "="*70)
print("EVALUATION REPORT")
print("="*70)
print(f"\nTest Samples: {len(all_preds)}")

print("\n📊 SYSTOLIC BP:")
print(f"   MAE:  {mae_sys:.2f} mmHg")
print(f"   RMSE: {rmse_sys:.2f} mmHg")
print(f"   Correlation: {corr_sys:.3f} (p={p_sys:.4f})")

print("\n📊 DIASTOLIC BP:")
print(f"   MAE:  {mae_dia:.2f} mmHg")
print(f"   RMSE: {rmse_dia:.2f} mmHg")
print(f"   Correlation: {corr_dia:.3f} (p={p_dia:.4f})")

print("\n🎯 CLINICAL ASSESSMENT:")
avg_mae = (mae_sys + mae_dia) / 2
if avg_mae < 5:
    print(f"   ✅ EXCELLENT (Avg MAE: {avg_mae:.2f} < 5 mmHg)")
elif avg_mae < 8:
    print(f"   ✅ GOOD (Avg MAE: {avg_mae:.2f} < 8 mmHg)")
elif avg_mae < 15:
    print(f"   ⚠️  ACCEPTABLE (Avg MAE: {avg_mae:.2f} < 15 mmHg)")
else:
    print(f"   ❌ NEEDS IMPROVEMENT (Avg MAE: {avg_mae:.2f} mmHg)")

# Create comprehensive plots
fig = plt.figure(figsize=(16, 12))

# 1. Bland-Altman Plot - Systolic
ax1 = plt.subplot(3, 3, 1)
mean_sys = (all_true[:, 0] + all_preds[:, 0]) / 2
diff_sys = all_true[:, 0] - all_preds[:, 0]
md_sys = np.mean(diff_sys)
sd_sys = np.std(diff_sys)

ax1.scatter(mean_sys, diff_sys, alpha=0.5, s=30)
ax1.axhline(md_sys, color='red', linestyle='--', label=f'Mean: {md_sys:.2f}')
ax1.axhline(md_sys + 1.96*sd_sys, color='gray', linestyle='--', label=f'+1.96 SD: {md_sys + 1.96*sd_sys:.2f}')
ax1.axhline(md_sys - 1.96*sd_sys, color='gray', linestyle='--', label=f'-1.96 SD: {md_sys - 1.96*sd_sys:.2f}')
ax1.set_xlabel('Mean Systolic BP (mmHg)')
ax1.set_ylabel('Difference (True - Pred)')
ax1.set_title('Bland-Altman: Systolic', fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. Bland-Altman Plot - Diastolic
ax2 = plt.subplot(3, 3, 2)
mean_dia = (all_true[:, 1] + all_preds[:, 1]) / 2
diff_dia = all_true[:, 1] - all_preds[:, 1]
md_dia = np.mean(diff_dia)
sd_dia = np.std(diff_dia)

ax2.scatter(mean_dia, diff_dia, alpha=0.5, s=30, color='coral')
ax2.axhline(md_dia, color='red', linestyle='--', label=f'Mean: {md_dia:.2f}')
ax2.axhline(md_dia + 1.96*sd_dia, color='gray', linestyle='--', label=f'+1.96 SD: {md_dia + 1.96*sd_dia:.2f}')
ax2.axhline(md_dia - 1.96*sd_dia, color='gray', linestyle='--', label=f'-1.96 SD: {md_dia - 1.96*sd_dia:.2f}')
ax2.set_xlabel('Mean Diastolic BP (mmHg)')
ax2.set_ylabel('Difference (True - Pred)')
ax2.set_title('Bland-Altman: Diastolic', fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# 3. Scatter Plot - Systolic
ax3 = plt.subplot(3, 3, 3)
ax3.scatter(all_true[:, 0], all_preds[:, 0], alpha=0.5, s=30)
ax3.plot([80, 200], [80, 200], 'r--', linewidth=2, label='Perfect')
ax3.set_xlabel('True Systolic (mmHg)')
ax3.set_ylabel('Predicted Systolic (mmHg)')
ax3.set_title(f'Systolic Predictions (r={corr_sys:.3f})', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Scatter Plot - Diastolic
ax4 = plt.subplot(3, 3, 4)
ax4.scatter(all_true[:, 1], all_preds[:, 1], alpha=0.5, s=30, color='coral')
ax4.plot([40, 120], [40, 120], 'r--', linewidth=2, label='Perfect')
ax4.set_xlabel('True Diastolic (mmHg)')
ax4.set_ylabel('Predicted Diastolic (mmHg)')
ax4.set_title(f'Diastolic Predictions (r={corr_dia:.3f})', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Error Distribution - Systolic
ax5 = plt.subplot(3, 3, 5)
ax5.hist(sys_error, bins=30, edgecolor='black', alpha=0.7)
ax5.axvline(0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Error (mmHg)')
ax5.set_ylabel('Frequency')
ax5.set_title(f'Systolic Error Distribution (MAE={mae_sys:.2f})', fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Error Distribution - Diastolic
ax6 = plt.subplot(3, 3, 6)
ax6.hist(dia_error, bins=30, edgecolor='black', alpha=0.7, color='coral')
ax6.axvline(0, color='red', linestyle='--', linewidth=2)
ax6.set_xlabel('Error (mmHg)')
ax6.set_ylabel('Frequency')
ax6.set_title(f'Diastolic Error Distribution (MAE={mae_dia:.2f})', fontweight='bold')
ax6.grid(True, alpha=0.3)

# 7. Residual Plot - Systolic
ax7 = plt.subplot(3, 3, 7)
ax7.scatter(all_preds[:, 0], sys_error, alpha=0.5, s=30)
ax7.axhline(0, color='red', linestyle='--', linewidth=2)
ax7.set_xlabel('Predicted Systolic (mmHg)')
ax7.set_ylabel('Residual (mmHg)')
ax7.set_title('Systolic Residuals', fontweight='bold')
ax7.grid(True, alpha=0.3)

# 8. Residual Plot - Diastolic
ax8 = plt.subplot(3, 3, 8)
ax8.scatter(all_preds[:, 1], dia_error, alpha=0.5, s=30, color='coral')
ax8.axhline(0, color='red', linestyle='--', linewidth=2)
ax8.set_xlabel('Predicted Diastolic (mmHg)')
ax8.set_ylabel('Residual (mmHg)')
ax8.set_title('Diastolic Residuals', fontweight='bold')
ax8.grid(True, alpha=0.3)

# 9. Performance Summary Table
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

summary_text = f"""
PERFORMANCE SUMMARY

Systolic BP:
  MAE:  {mae_sys:.2f} mmHg
  RMSE: {rmse_sys:.2f} mmHg
  r:    {corr_sys:.3f}

Diastolic BP:
  MAE:  {mae_dia:.2f} mmHg
  RMSE: {rmse_dia:.2f} mmHg
  r:    {corr_dia:.3f}

Overall:
  Avg MAE: {avg_mae:.2f} mmHg
  
Clinical Grade:
  {"✅ EXCELLENT" if avg_mae < 5 else "✅ GOOD" if avg_mae < 8 else "⚠️ ACCEPTABLE"}
"""

ax9.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('results/comprehensive_evaluation.png', dpi=200, bbox_inches='tight')
print("\n✅ Comprehensive evaluation saved to results/comprehensive_evaluation.png")
plt.show()

print("="*70)