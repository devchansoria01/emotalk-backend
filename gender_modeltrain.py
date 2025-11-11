import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from gender_model import GenderClassifier
import os
import numpy as np

# --- Configuration ---
DATA_DIR = r'C:\Users\anike\OneDrive\Desktop\ds project\HumanPatterns\Validation' # **IMPORTANT: Change this to your dataset's parent directory name**
BATCH_SIZE = 64
IMAGE_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
MODEL_SAVE_PATH = 'gender_classifier_best.pth'

# Check for GPU availability
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model():
    """
    Main function to load data, initialize model, and run the training loop.
    """
    print(f"--- Starting Training on Device: {DEVICE} ---")
    
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory '{DATA_DIR}' not found.")
        print("Please ensure your data is structured as: "
              f"'{DATA_DIR}/male/' and '{DATA_DIR}/female/'")
        return

    # --- 1. Data Loading and Preprocessing ---
    
    # Define transformations for the images
    data_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Resize all images to 64x64
        transforms.ToTensor(), # Convert PIL Image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the entire dataset using ImageFolder
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=data_transforms)
    
    # Get class names (should be ['female', 'male'] or vice versa based on folder order)
    class_names = full_dataset.classes
    print(f"Found {len(full_dataset)} images belonging to {len(class_names)} classes: {class_names}")

    # Split the dataset into training and validation sets (e.g., 80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    # --- 2. Model, Loss, and Optimizer Initialization ---
    
    model = GenderClassifier(num_classes=len(class_names)).to(DEVICE)
    
    # CrossEntropyLoss is suitable for multi-class classification (and binary)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training Loop ---
    
    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train() # Set model to training mode
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_dataset)

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        correct = 0
        total = 0
        
        with torch.no_grad(): # Disable gradient calculation for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                
                # Get the predicted class (index of the highest logit)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Validation Accuracy: {val_accuracy:.2f}%")

        # --- Save Best Model ---
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> Model saved to {MODEL_SAVE_PATH} with improved accuracy: {best_accuracy:.2f}%")

    print("\n--- Training Complete ---")
    print(f"Best Validation Accuracy achieved: {best_accuracy:.2f}%")
    print(f"Final model saved at: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    train_model()