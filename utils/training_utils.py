import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def train_model(model, x_train, y_train, x_test, y_test, save_path, epochs=25, batch_size=20, device='cuda'):
    # Move model to device (GPU or CPU)
    model.to(device)
    
    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    # Training loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            
            # Compute accuracy
            train_pred = (outputs >= 0.5).float()
            val_pred = (val_outputs >= 0.5).float()
            
            train_accuracy = (train_pred == y_train_tensor).float().mean().item()
            val_accuracy = (val_pred == y_test_tensor).float().mean().item()
        
        # Save metrics
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f"{save_path}_best_model.pth")
        
        # Update learning rate
        scheduler.step()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, save_path):
    # Plot loss
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{save_path}/loss_plot.png")
    plt.show()
    
    # Plot accuracy
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{save_path}/accuracy_plot.png")
    plt.show()

def evaluate_model(model, x_test, y_test, device='cuda'):
    model.eval()
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        outputs = model(x_test_tensor)
        y_pred = (outputs >= 0.5).float().cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()
    
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))