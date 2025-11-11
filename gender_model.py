import torch
import torch.nn as nn
import torch.nn.functional as F

# A simple Convolutional Neural Network (CNN) for image classification.
# It's designed to take a 3-channel (RGB) image and classify it into one of 2 classes (male/female).

class GenderClassifier(nn.Module):
    """
    CNN model for binary gender classification.
    """
    def __init__(self, num_classes=2):
        super(GenderClassifier, self).__init__()
        
        # --- Feature Extraction Layers ---
        # Input image size will be 64x64x3 (after transformation)
        
        # 1st Convolutional Block
        # 3 input channels (RGB), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Output size: 64x64x32
        
        # 2nd Convolutional Block
        # 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Output size: 32x32x64 (after pooling)
        
        # 3rd Convolutional Block
        # 64 input channels, 128 output channels, 3x3 kernel
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Output size: 16x16x128 (after pooling)
        
        # Max Pooling layer (applied after each conv layer)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # --- Fully Connected Layers ---
        # The size calculation below assumes input images are resized to 64x64.
        # After conv3 and two pool layers, the feature map size is 8x8x128.
        # 64 -> 32 (pool 1) -> 16 (pool 2) -> 8 (pool 3)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

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
        
        # Final output layer (no softmax/sigmoid here, as CrossEntropyLoss handles it)
        x = self.fc2(x)
        return x