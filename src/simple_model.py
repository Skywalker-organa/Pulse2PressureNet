# src/simple_model.py - Simple CNN Model
import torch
import torch.nn as nn

class SimpleBPModel(nn.Module):
    """
    Simple CNN to predict Blood Pressure from PPG
    """
    def __init__(self):
        super(SimpleBPModel, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=10,
            stride=2
        )
        self.pool1 = nn.MaxPool1d(2)

        # Layer 2
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=2
        )
        self.pool2 = nn.MaxPool1d(2)

        # After conv layers → feature length becomes 76
        self.fc1 = nn.Linear(64 * 76, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Test the model
if __name__ == "__main__":
    print("Testing model...")

    model = SimpleBPModel()

    total_params = sum(p.numel() for p in model.parameters())
    print("Model created!")
    print(f"   Parameters: {total_params:,}")

    dummy_input = torch.randn(4, 1, 1250)
    output = model(dummy_input)

    print("\nForward pass works!")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Sample prediction: [{output[0,0]:.1f}, {output[0,1]:.1f}]")
