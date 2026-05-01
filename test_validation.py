import cv2
import numpy as np
import sys
from tensorflow.keras.preprocessing import image
try:
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
    mobilenet_model = MobileNetV2(weights='imagenet')
except Exception as e:
    print(f"Could not load MobileNetV2: {e}")
    mobilenet_model = None

def is_valid_leaf_image(filepath):
    img_cv = cv2.imread(filepath)
    if img_cv is None:
        return False
        
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            print("Face detected!")
            return False 
    except:
        pass
        
    if mobilenet_model is not None:
        try:
            img = image.load_img(filepath, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = mobilenet_preprocess(x)
            
            preds = mobilenet_model.predict(x, verbose=0)
            decoded = decode_predictions(preds, top=5)[0]
            
            print("MobileNet Predictions:")
            for _, label, prob in decoded:
                print(f" - {label}: {prob:.2f}")
                
            nature_keywords = [
                'plant', 'leaf', 'flower', 'tree', 'fruit', 'vegetable', 'wood', 'grass', 
                'pot', 'daisy', 'corn', 'acorn', 'squash', 'cucumber', 'artichoke', 'pepper', 
                'cardoon', 'mushroom', 'apple', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 
                'banana', 'jackfruit', 'pomegranate', 'hay', 'greenhouse', 'buckeye', 'earthstar', 
                'bolete', 'ear', 'beetle', 'bug', 'snail', 'slug', 'worm', 'ant', 'fly', 'bee', 'mantis',
                'grasshopper', 'spider', 'web', 'nematode', 'earthworm', 'ladybug', 'butterfly', 'moth',
                'weevil', 'lacewing', 'dragonfly', 'damselfly', 'soil', 'ground', 'dirt', 'stone',
                'rock', 'pebble', 'sand', 'mud', 'vase', 'basket', 'pitcher', 'vine', 'fern', 'moss',
                'rapeseed', 'broccoli', 'cauliflower', 'zucchini', 'macaw', 'lorikeet'
            ]
            
            is_nature_like = False
            for _, label, prob in decoded:
                label_lower = label.lower().replace('_', ' ')
                if any(kw in label_lower for kw in nature_keywords):
                    is_nature_like = True
                    print(f"Matched nature keyword in: {label_lower}")
                    break
                    
            if not is_nature_like:
                hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array([10, 20, 20]), np.array([100, 255, 255]))
                ratio = cv2.countNonZero(mask) / (img_cv.shape[0] * img_cv.shape[1])
                print(f"Not nature like. Color ratio: {ratio:.2f}")
                if ratio < 0.15:
                    return False
        except Exception as e:
            print(f"Validation error: {e}")
            pass
            
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        res = is_valid_leaf_image(sys.argv[1])
        print(f"Is valid leaf: {res}")
