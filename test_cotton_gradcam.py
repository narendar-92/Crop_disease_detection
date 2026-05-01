import os
import cv2
import numpy as np
import tensorflow as tf
from app import make_gradcam_heatmap, prepare_image, preprocess_for_crop

# Load the model
try:
    model = tf.keras.models.load_model('models/cotton_model.keras')
    print("Cotton Model loaded successfully.")
except Exception as e:
    print("Failed to load model:", e)
    exit(1)

# Create a dummy image
img_path = 'test_dummy_cotton.jpg'
cv2.imwrite(img_path, np.zeros((224, 224, 3), dtype=np.uint8))

img_array = preprocess_for_crop(prepare_image(img_path), "Cotton")

try:
    preds = model.predict(img_array)
    class_index = int(np.argmax(preds[0]))
    print(f"Prediction index: {class_index}")
    
    heatmap = make_gradcam_heatmap(img_array, model, class_index, layer_name=None)
    print("Grad-CAM generation successful. Heatmap shape:", heatmap.shape)
except Exception as e:
    import traceback
    print("Grad-CAM failed:", e)
    traceback.print_exc()

os.remove(img_path)
