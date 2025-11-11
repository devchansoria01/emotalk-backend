import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import os
import sys

# --- Import your model architecture ---
# Ensure 'age_model.py' is in the same directory
try:
    from age_model import AgeRegressor
except ImportError:
    print("Error: Could not import AgeRegressor from 'age_model.py'.")
    print("Please ensure 'age_model.py' is in the current directory.")
    sys.exit(1)

# --- Configuration ---
MODEL_PATH = 'age_regressor_best.pth'
IMAGE_SIZE = 64
HAARCASCADE_PATH = 'haarcascade_frontalface_default.xml' # IMPORTANT: Update this path!

# Check for GPU (though CPU is usually faster for single-frame inference)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the image transformations (must match the training script)
data_transforms = transforms.Compose([
    transforms.ToPILImage(), # Convert NumPy array (from CV2) to PIL Image
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # Normalization parameters must match the training script
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    """Loads the trained PyTorch model and sets it to evaluation mode."""
    print(f"Loading model on device: {DEVICE}...")
    
    # Initialize the model
    model = AgeRegressor().to(DEVICE)
    
    # Load the state dictionary
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Please ensure you have trained the model using 'age_modeltrain.py' first.")
        return None

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set model to inference mode
    
    print("Model loaded successfully.")
    return model

def main():
    """Initializes model, face detector, and runs the video processing loop."""
    
    # 1. Load the PyTorch Model
    model = load_model()
    if model is None:
        return

    # 2. Initialize the Face Detector (Haar Cascade)
    if not os.path.exists(HAARCASCADE_PATH):
        print(f"\nWARNING: Haar Cascade file not found at {HAARCASCADE_PATH}")
        print("Please download 'haarcascade_frontalface_default.xml' and place it in the same directory, or update the path.")
        print("Running without face detection (processing entire frame, results will be inaccurate).")
        face_cascade = None
    else:
        face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)

    # 3. Initialize Video Capture
    cap = cv2.VideoCapture(0) # 0 is the default camera
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("\n--- Running Inference. Press 'q' to exit. ---")
    
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection (faster)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Color for drawing (BGR)
        color = (0, 255, 0) # Green

        if face_cascade:
            # Detect faces
            # The last two parameters adjust scale factor and minimum neighbors
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # 4. Extract and Preprocess the Face
                
                # Expand the bounding box slightly for better context
                margin = int(0.2 * w)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                
                face_img = frame[y1:y2, x1:x2]
                
                # Apply the standard data transformations
                try:
                    input_tensor = data_transforms(face_img).unsqueeze(0).to(DEVICE)
                except Exception as e:
                    # Skip frame if transformation fails (e.g., face too small)
                    print(f"Skipping frame due to transformation error: {e}")
                    continue

                # 5. Model Inference
                with torch.no_grad():
                    output = model(input_tensor)
                    
                    # The model output is a single continuous age prediction (tensor of shape [1, 1])
                    predicted_age = output.item()
                
                # 6. Display Result
                age_text = f"Age: {predicted_age:.1f}"
                
                # Draw bounding box and text
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, age_text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        else:
            # Placeholder if face detection is not available (for debugging)
            cv2.putText(frame, "Face Detector Missing!", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)


        # Display the resulting frame
        cv2.imshow('Age Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()