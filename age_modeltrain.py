import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
# Assuming the model file is saved as 'age_model.py'
from age_model import AgeRegressor 
import os
import numpy as np

# *** IMPORTANT: You will need to create an AgeDataset class ***
# This class needs to load image paths and their corresponding age labels (e.g., from a CSV).
# Since I cannot see your data, I'll use a placeholder/dummy class for demonstration.
class PlaceholderAgeDataset(Dataset):
    """
    A placeholder for a real dataset that loads images and their continuous age labels.
    In a real scenario, this would load a CSV mapping image files to age.
    """
    def __init__(self, data_transforms, num_samples=1000):
        # Dummy data generation: 1000 random tensors and 1000 random ages
        self.data = [torch.randn(3, 64, 64) for _ in range(num_samples)]
        # Ages between 1 and 100
        self.labels = [torch.tensor(np.random.randint(1, 100), dtype=torch.float32) for _ in range(num_samples)]
        self.transforms = data_transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # In a real implementation: load image from disk, apply transforms, get age label
        image = self.data[idx]
        label = self.labels[idx]
        return image, label.unsqueeze(0) # unsqueeze(0) for shape [1]

# --- Configuration (UPDATED) ---
# DATA_DIR is less relevant for custom dataset, but kept for context.
DATA_DIR = r'C:\Users\anike\OneDrive\Desktop\ds project\HumanPatterns\agedetectiondataset\train' 
BATCH_SIZE = 64
IMAGE_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
MODEL_SAVE_PATH = 'age_regressor_best.pth'

# Check for GPU availability
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model():
    """
    Main function to load data, initialize model, and run the training loop for AGE REGRESSION.
    """
    print(f"--- Starting Age Regression Training on Device: {DEVICE} ---")
    
    # --- 1. Data Loading and Preprocessing ---
    
    # Define transformations for the images
    data_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the entire dataset using the custom/placeholder AgeDataset
    full_dataset = PlaceholderAgeDataset(data_transforms) 
    print(f"Found {len(full_dataset)} total samples.")

    # Split the dataset (e.g., 80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders
    # Note: num_workers=4 might cause issues on some Windows setups, use 0 if necessary.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    # --- 2. Model, Loss, and Optimizer Initialization (UPDATED) ---
    
    # Initialize the AgeRegressor model
    model = AgeRegressor().to(DEVICE)
    
    # Mean Squared Error Loss (MSELoss) is standard for regression
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training Loop ---
    
    best_loss = float('inf') # Track best validation LOSS

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            # IMPORTANT: Labels are now continuous floats for regression
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE) 

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Loss calculation for regression
            loss = criterion(outputs, labels) 

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss_total = 0.0
        # For regression, we often track Mean Absolute Error (MAE) as an understandable metric
        mae_total = 0.0 
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                
                # Calculate validation loss (MSE)
                val_loss_total += criterion(outputs, labels).item() * inputs.size(0)
                
                # Calculate Mean Absolute Error (MAE): |predicted - true|
                mae_total += torch.sum(torch.abs(outputs - labels)).item()

        val_loss = val_loss_total / len(val_dataset)
        val_mae = mae_total / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss (MSE): {train_loss:.4f} | "
              f"Validation Loss (MSE): {val_loss:.4f} | "
              f"Validation MAE: {val_mae:.2f}") # MAE is the average age error in years

        # --- Save Best Model ---
        # Saving the model with the lowest validation loss (MSE)
        if val_loss < best_loss: 
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> Model saved to {MODEL_SAVE_PATH} with improved Val Loss: {best_loss:.4f}")

    print("\n--- Training Complete ---")
    print(f"Best Validation Loss (MSE) achieved: {best_loss:.4f}")
    print(f"Final model saved at: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    train_model()