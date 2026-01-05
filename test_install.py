# test_install.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("✅ PyTorch version:", torch.__version__)
print("✅ NumPy version:", np.__version__)
print("✅ Pandas version:", pd.__version__)
print("✅ All libraries installed successfully!")

# Test GPU (optional)
if torch.cuda.is_available():
    print("🚀 GPU available:", torch.cuda.get_device_name(0))
else:
    print("💻 Using CPU (this is fine for learning)")