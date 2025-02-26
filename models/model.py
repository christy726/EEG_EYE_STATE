import torch
import torch.nn as nn

class EEGModel(nn.Module):
    def __init__(self):
        super(EEGModel, self).__init__()
        self.dense1 = nn.Linear(14, 64)
        self.lstm1 = nn.LSTM(64, 256, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(512, 128, batch_first=True, bidirectional=True)
        self.flatten = nn.Flatten()
        self.dense2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.dense1(x))
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.flatten(x)
        x = self.relu(self.dense2(x))
        x = self.sigmoid(self.output(x))
        return x
