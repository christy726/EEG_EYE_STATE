import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Check for missing values
    if df.isnull().sum().any():
        raise ValueError("Dataset contains missing values.")
    
    # Separate features and target
    y = df.pop('eyeDetection')
    x = df
    
    # Standardize features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled, columns=x.columns)
    
    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Reshape for LSTM input (PyTorch expects [batch_size, sequence_length, input_size])
    x_train = x_train.values.reshape(-1, 14, 1)
    x_test = x_test.values.reshape(-1, 14, 1)
    
    return x_train, x_test, y_train, y_test