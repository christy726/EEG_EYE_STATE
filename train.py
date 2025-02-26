import torch
import torch.nn as nn
import torch.optim as optim
from models.model import EEGModel
from utils.data_processing import load_data, preprocess_data, create_dataloaders
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def train_model(model, train_loader, test_loader, epochs=25, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            preds = (outputs >= 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

        train_accuracy = correct_predictions / total_predictions
        history['loss'].append(running_loss / len(train_loader))
        history['accuracy'].append(train_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {train_accuracy}")

        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (outputs >= 0.5).float()
                val_correct_predictions += (preds == labels).sum().item()
                val_total_predictions += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_accuracy = val_correct_predictions / val_total_predictions
        history['val_loss'].append(val_loss / len(test_loader))
        history['val_accuracy'].append(val_accuracy)

        print(f"Validation Loss: {val_loss/len(test_loader)}, Validation Accuracy: {val_accuracy}")
        print(confusion_matrix(all_labels, all_preds))
        print(classification_report(all_labels, all_preds))

    plot_metrics(history)
    return model

def plot_metrics(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.show()

if __name__ == "__main__":
    df = load_data("data/EEG_Eye_State_Classification.csv")
    x, y = preprocess_data(df)
    train_loader, test_loader = create_dataloaders(x, y)
    model = EEGModel()
    trained_model = train_model(model, train_loader, test_loader)
    torch.save(trained_model.state_dict(), "models/eeg_model.pth")