import os
import numpy as np
from utils.data_preprocessing import load_and_preprocess_data
from utils.model_architecture import EEGModel
from utils.training_utils import train_model, plot_training_history, evaluate_model

# Set paths
DATA_PATH = "data/EEG_Eye_State_Classification.csv"
MODEL_SAVE_PATH = "models/"
PLOTS_PATH = "plots/"

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

# Load and preprocess data
x_train, x_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)

# Build model
model = EEGModel(input_size=1, hidden_size=128, num_layers=2, dropout=0.3)

# Train model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, x_train, y_train, x_test, y_test, MODEL_SAVE_PATH, epochs=25, batch_size=20, device=device
)

# Plot training history
plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, PLOTS_PATH)

# Evaluate model
evaluate_model(model, x_test, y_test, device=device)