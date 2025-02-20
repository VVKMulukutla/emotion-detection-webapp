import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the ResNet50 model from the .h5 checkpoint
model = tf.keras.models.load_model("/home/vamsikrishnamulukutla/Downloads/model_checkpoints/resnet50_emotion.h5")


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # ResNet50 expects 224x224 images
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1] (assuming your model expects this)
    return img_array


def infer_resnet50(img_path):
    # Preprocess the image
    img_array = preprocess_image(img_path)

    # Perform inference
    predictions = model.predict(img_array)

    # Assuming your model outputs probabilities for emotion classes
    emotion_classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise",
                       "Neutral"]  # Replace with your actual classes
    predicted_class = emotion_classes[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_class, confidence


# Example usage
if __name__ == "__main__":
    image_path = "/home/vamsikrishnamulukutla/Downloads/imagesForFER/disgust/image.jpg"  # Replace with the path to your image
    predicted_class, confidence = infer_resnet50(image_path)

    print(f"Predicted Emotion: {predicted_class} (Confidence: {confidence:.2f})")