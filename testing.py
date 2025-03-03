import joblib
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms

# 1. Load the saved ResNet-50 model using Joblib
model = joblib.load('model_checkpoints/resnet_joblib.joblib')  # Replace with your model file path
model.eval()  # Set model to evaluation mode (important for inference)


# 2. Load and preprocess the image
def load_and_preprocess_image(image_path):
    # Define PyTorch preprocessing pipeline (matches ResNet-50 ImageNet training)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),  # Convert to tensor (0-1 range, CHW format)
        transforms.Normalize(  # Normalize with ImageNet mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Open image and apply preprocessing
    img = Image.open(image_path).convert('RGB')  # Ensure 3 channels (RGB)
    img_tensor = preprocess(img)  # Apply transforms
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)

    return img_tensor


# 3. Path to your local image
image_path = 'test_images/Training_10218600.jpg'  # Replace with your image path
processed_image = load_and_preprocess_image(image_path)

# 4. Predict the class (move to GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
processed_image = processed_image.to(device)

with torch.no_grad():  # Disable gradient computation for inference
    predictions = model(processed_image)  # Get model output (logits)

# 5. Interpret the predictions
# Convert logits to probabilities (optional, depending on model output)
probabilities = torch.softmax(predictions, dim=1)  # Apply softmax if logits
predicted_class = torch.argmax(probabilities, dim=1)  # Get class index
print(f"Predicted class index: {predicted_class.item()}")

# If using ImageNet classes, decode them
try:
    # Load ImageNet class labels (you’ll need a labels file or download one)
    import json

    with open('imagenet_labels.json', 'r') as f:  # Replace with your labels file
        imagenet_labels = json.load(f)

    predicted_idx = predicted_class.item()
    predicted_label = imagenet_labels[str(predicted_idx)]
    confidence = probabilities[0, predicted_idx].item()
    print(f"Predicted class: {predicted_label} (confidence: {confidence:.2f})")
except:
    print("Decoding not available; use your own class labels for mapping.")