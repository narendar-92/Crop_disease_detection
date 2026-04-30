import os
import cv2
import numpy as np
import tensorflow as tf
from app import make_gradcam_heatmap, mixed_classes, prepare_image, preprocess_for_crop

# Load the model
try:
    model = tf.keras.models.load_model('models/crop_disease_model.keras')
    print("Model loaded successfully.")
except Exception as e:
    print("Failed to load model:", e)
    exit(1)

# Create a dummy image
img_path = 'test_dummy.jpg'
cv2.imwrite(img_path, np.zeros((224, 224, 3), dtype=np.uint8))

img_array = preprocess_for_crop(prepare_image(img_path), "Pepper")

try:
    preds = model.predict(img_array)
    class_index = int(np.argmax(preds[0]))
    print(f"Prediction: {mixed_classes[class_index]}")
    
    heatmap = make_gradcam_heatmap(img_array, model, class_index)
    print("Grad-CAM generation successful. Heatmap shape:", heatmap.shape)
except Exception as e:
    import traceback
    print("Grad-CAM failed:", e)
    traceback.print_exc()

os.remove(img_path)
