import torch
import torch.nn as nn

class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Model, self).__init__()
        
        # Part 1: CNN layers (extract spatial features)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=10, stride=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        
        # Part 2: LSTM layers (capture temporal patterns)
        self.lstm = nn.LSTM(
            input_size=128,      # Features from CNN
            hidden_size=64,      # LSTM memory size
            num_layers=2,        # Stack 2 LSTMs
            batch_first=True,    # Input shape: (batch, time, features)
            dropout=0.3          # Regularization
        )
        
        # Part 3: Fully connected (final decision)
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 2)  # Output: [systolic, diastolic]
    
    def forward(self, x):
        """Input: PPG signal (batch, 1, 1250)
        Output: BP prediction (batch, 2)"""
        # CNN feature extraction
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        
        # Reshape for LSTM: (batch, channels, time) → (batch, time, channels)
        x = x.permute(0, 2, 1)
        
        # LSTM processes sequence
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        x = h_n[-1]  # (batch, 64)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# Test it
if __name__ == "__main__":
    print("Testing CNN-LSTM model")
    
    model = CNN_LSTM_Model()
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"Model created")
    print(f"   Parameters: {total:,}")
    
    # Test forward pass
    test_input = torch.randn(4, 1, 1250)
    output = model(test_input)
    
    print("Forward pass works")
    print(f"   Input: {test_input.shape}")
    print(f"   Output: {output.shape}")