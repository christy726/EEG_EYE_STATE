import torch
import torch.nn as nn

class EEGModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.3):
        super(EEGModel, self).__init__()
        
        # Dense layer
        self.dense1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 128)  # Multiply by 2 for bidirectional
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Dense layer
        x = self.relu(self.dense1(x))
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Use only the last time step
        
        # Fully connected layers
        out = self.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        
        return out