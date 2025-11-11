import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from gender_model import GenderClassifier
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = 'gender_classifier_best.pth'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
IMAGE_SIZE = 64 # Must match the size used in main.py
CLASS_NAMES = ['female', 'male'] # Must match class order found by ImageFolder in main.py

# Check for GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """
    Loads the trained PyTorch model weights into the defined architecture.
    """
    try:
        # Initialize the model structure
        model = GenderClassifier(num_classes=len(CLASS_NAMES))
        
        # Load the saved state dictionary
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at {MODEL_PATH}")
            print("Please run main.py first to train and save the model.")
            return None

        # Load weights and move model to device
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() # Set model to evaluation mode
        print(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def predict_gender(model, face_img):
    """
    Preprocesses a single face image and predicts the gender.
    """
    # 1. Define the exact transformations used during training (Normalization is key!)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Apply preprocessing
    input_tensor = preprocess(face_img)
    
    # 3. Add batch dimension (C x H x W) -> (1 x C x H x W)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)
    
    # 4. Make prediction
    with torch.no_grad():
        output = model(input_batch)
        
    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=1)
    
    # Get the predicted class index and confidence
    confidence, predicted_class_idx = torch.max(probabilities, 1)
    
    # Extract the name and formatted confidence
    predicted_name = CLASS_NAMES[predicted_class_idx.item()]
    confidence_score = confidence.item() * 100
    
    return predicted_name, confidence_score

def run_webcam_detection():
    """
    Opens the webcam, detects faces, and displays gender predictions.
    """
    model = load_model()
    if model is None:
        return

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE_PATH)
    if face_cascade.empty():
        print(f"ERROR: Failed to load Haar Cascade file at {cv2.data.haarcascades + CASCADE_PATH}")
        print("Ensure 'haarcascade_frontalface_default.xml' is accessible or installed correctly.")
        return

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to grayscale for cascade detection (faster)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )

            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw bounding box (blue)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Extract the face region (crop using numpy slicing)
                face_crop = frame[y:y+h, x:x+w]
                
                # Convert BGR (OpenCV format) to RGB (PyTorch/PIL format expected by ToPILImage)
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                
                # Predict gender
                gender, confidence = predict_gender(model, face_rgb)
                
                # Prepare text label
                label = f"{gender.capitalize()} ({confidence:.1f}%)"
                
                # Display the label above the bounding box (green text)
                cv2.putText(frame, label, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Gender Prediction (Press Q to quit)', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # When everything done, release the capture and destroy windows
        cap.release()
        cv2.destroyAllWindows()
        print("\nWebcam session ended.")

if __name__ == '__main__':
    run_webcam_detection()