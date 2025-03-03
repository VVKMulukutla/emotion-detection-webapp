import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("model_checkpoints/resnet50_emotion.h5")


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def infer_resnet50(img_path):

    img_array = preprocess_image(img_path)

    predictions = model.predict(img_array)

    emotion_classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise",
                       "Neutral"]
    predicted_class = emotion_classes[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_class, confidence



if __name__ == "__main__":
    image_path = "/home/vamsikrishnamulukutla/Downloads/imagesForFER/disgust/image.jpg"  # Replace with the path to your image
    predicted_class, confidence = infer_resnet50(image_path)

    print(f"Predicted Emotion: {predicted_class} (Confidence: {confidence:.2f})")