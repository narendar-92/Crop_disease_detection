import sys
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

def is_leaf(img_path):
    model = MobileNetV2(weights='imagenet')
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=5)[0]
    
    print("Predictions for", img_path)
    for i, (imagenet_id, label, prob) in enumerate(decoded):
        print(f"{i + 1}: {label} ({prob * 100:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        is_leaf(sys.argv[1])
