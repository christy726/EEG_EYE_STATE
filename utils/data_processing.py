import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    data = df.copy()
    y = data.pop('eyeDetection')
    x = data
    x_new = StandardScaler().fit_transform(x)
    x_new = pd.DataFrame(x_new, columns=x.columns)
    return x_new, y

def create_dataloaders(x, y, batch_size=32, test_size=0.15):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train = torch.tensor(x_train.values, dtype=torch.float32).view(-1, 14, 1)
    x_test = torch.tensor(x_test.values, dtype=torch.float32).view(-1, 14, 1)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
