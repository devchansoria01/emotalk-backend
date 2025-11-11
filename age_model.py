import torch
import torch.nn as nn
import torch.nn.functional as F

# A simple Convolutional Neural Network (CNN) for age regression.
# It's designed to take a 3-channel (RGB) image and predict a single age value.

class AgeRegressor(nn.Module):
    """
    CNN model for age regression.
    """
    def __init__(self):
        super(AgeRegressor, self).__init__()
        
        # --- Feature Extraction Layers (Kept the same structure) ---
        # Input image size will be 64x64x3 (after transformation)
        
        # 1st Convolutional Block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Output size: 64x64x32
        
        # 2nd Convolutional Block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Output size: 32x32x64 (after pooling)
        
        # 3rd Convolutional Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Output size: 16x16x128 (after pooling)
        
        # Max Pooling layer (applied after each conv layer)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # --- Fully Connected Layers ---
        # After conv3 and three pool layers (64 -> 32 -> 16 -> 8), the feature map size is 8x8x128.
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        
        # **Modified for Age Regression:** Output is a single value (the predicted age).
        # No activation function is applied to the final layer's output.
        self.fc2 = nn.Linear(512, 1) 

    def forward(self, x):
        # Apply 1st conv, ReLU, and pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply 2nd conv, ReLU, and pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Apply 3rd conv, ReLU, and pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 128 * 8 * 8)
        
        # Apply dropout and 1st fully connected layer
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        
        # Final output layer for regression (single predicted age)
        x = self.fc2(x)
        return x