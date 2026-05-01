import os
import uuid
import numpy as np
import cv2

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
    TENSORFLOW_AVAILABLE = True
    try:
        print("Loading MobileNetV2 for image validation...")
        mobilenet_model = MobileNetV2(weights='imagenet')
    except Exception as e:
        print(f"Could not load MobileNetV2: {e}")
        mobilenet_model = None
except ImportError:
    print("TensorFlow not available. Running in demo mode.")
    TENSORFLOW_AVAILABLE = False
    mobilenet_model = None

from flask import Flask, render_template, request, url_for, session, redirect
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["SECRET_KEY"] = "your-secret-key-change-in-production"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Limit uploads to 16MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# ===============================
# Load Models (with fallback for demo)
# ===============================

try:
    corn_model = load_model("models/corn_model.h5")
    paddy_model = load_model("models/paddy_model.h5")
    cotton_model = load_model("models/cotton_model.keras")
    mixed_model = load_model("models/crop_disease_model.keras")
    MODELS_LOADED = True
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    print("Running in demo mode with mock predictions")
    MODELS_LOADED = False
    corn_model = paddy_model = cotton_model = mixed_model = None

# ===============================
# Class Names
# ===============================

corn_classes = ["blight", "common_rust", "gray_leaf_spot", "healthy"]

paddy_classes = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
]

cotton_classes = ["bacterial_blight", "curl_virus", "fussarium_wilt", "healthy"]

mixed_classes = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

# ===============================
# Disease + Store Info
# ===============================

disease_info = {
    'Pepper__bell___Bacterial_spot': {'description': 'Small dark brown to black water-soaked spots appear on leaves.', 'cause': 'Specific pathogen related to Pepper  bell   Bacterial spot', 'pesticide': 'Spray copper oxychloride 2.5-3 g per liter of water. Rotate with copper hydroxide plus mancozeb where labels allow. Repeat spray at 7 day interval during active spread.', 'organic_alternatives': 'Spray neem oil 3-5 ml per liter with proper emulsifier. Use Bacillus-based bio-products if locally available. Remove badly infected leaves before organic spray.', 'prevention': 'Avoid overhead irrigation and keep foliage dry as much as possible.', 'spray_time': 'Do not spray before rainfall; restart protection after heavy rain washes deposits away.', 'severity': 'High', 'progression': 'Older leaves dry early and plant vigor starts dropping.', 'store_id': 'copper_oxychloride'},
    'Pepper__bell___healthy': {'description': 'Crop appears healthy.', 'cause': 'N/A', 'pesticide': 'No pesticide required.', 'organic_alternatives': 'N/A', 'prevention': 'Continue regular monitoring.', 'spray_time': 'N/A', 'severity': 'Safe', 'progression': 'Expected to remain healthy.', 'store_id': None},
    'Potato___Early_blight': {'description': 'Brown circular spots with concentric ring pattern appear first on older leaves.', 'cause': 'Specific pathogen related to Potato   Early blight', 'pesticide': 'Start protectant fungicide spray with mancozeb 2-2.5 g/L. Rotate with chlorothalonil or azoxystrobin as per local label. Maintain 7 day spray interval under strong disease pressure.', 'organic_alternatives': 'Spray neem extract or compost tea only after removing infected lower leaves. Mulch soil to reduce spore splash to new leaves. Improve airflow by pruning dense lower canopy.', 'prevention': 'Use furrow or drip irrigation and avoid evening leaf wetness.', 'spray_time': 'Warm humid weather increases infection, so maintain preventive cover before long cloudy spells.', 'severity': 'High', 'progression': 'Defoliation increases as the disease moves upward.', 'store_id': 'mancozeb'},
    'Potato___Late_blight': {'description': 'Large water-soaked lesions spread quickly on leaves.', 'cause': 'Specific pathogen related to Potato   Late blight', 'pesticide': 'Use systemic plus contact fungicide mix recommended for late blight. Common options include metalaxyl + mancozeb or cymoxanil + mancozeb as per label. Repeat at 5-7 day interval during cool wet periods.', 'organic_alternatives': 'Remove infected leaves immediately to lower inoculum. Improve drainage and reduce standing moisture around plants. Use copper-based organic-compatible sprays where permitted.', 'prevention': 'Stop overhead irrigation and keep field drainage open.', 'spray_time': 'Cool cloudy and wet weather can trigger explosive spread; maintain protective spray before such windows.', 'severity': 'High', 'progression': 'Leaves collapse rapidly and infection can move to stems and fruits or tubers.', 'store_id': 'copper_oxychloride'},
    'Potato___healthy': {'description': 'Crop appears healthy.', 'cause': 'N/A', 'pesticide': 'No pesticide required.', 'organic_alternatives': 'N/A', 'prevention': 'Continue regular monitoring.', 'spray_time': 'N/A', 'severity': 'Safe', 'progression': 'Expected to remain healthy.', 'store_id': None},
    'Tomato_Bacterial_spot': {'description': 'Small dark brown to black water-soaked spots appear on leaves.', 'cause': 'Specific pathogen related to Tomato Bacterial spot', 'pesticide': 'Spray copper oxychloride 2.5-3 g per liter of water. Rotate with copper hydroxide plus mancozeb where labels allow. Repeat spray at 7 day interval during active spread.', 'organic_alternatives': 'Spray neem oil 3-5 ml per liter with proper emulsifier. Use Bacillus-based bio-products if locally available. Remove badly infected leaves before organic spray.', 'prevention': 'Avoid overhead irrigation and keep foliage dry as much as possible.', 'spray_time': 'Do not spray before rainfall; restart protection after heavy rain washes deposits away.', 'severity': 'High', 'progression': 'Older leaves dry early and plant vigor starts dropping.', 'store_id': 'copper_oxychloride'},
    'Tomato_Early_blight': {'description': 'Brown circular spots with concentric ring pattern appear first on older leaves.', 'cause': 'Specific pathogen related to Tomato Early blight', 'pesticide': 'Start protectant fungicide spray with mancozeb 2-2.5 g/L. Rotate with chlorothalonil or azoxystrobin as per local label. Maintain 7 day spray interval under strong disease pressure.', 'organic_alternatives': 'Spray neem extract or compost tea only after removing infected lower leaves. Mulch soil to reduce spore splash to new leaves. Improve airflow by pruning dense lower canopy.', 'prevention': 'Use furrow or drip irrigation and avoid evening leaf wetness.', 'spray_time': 'Warm humid weather increases infection, so maintain preventive cover before long cloudy spells.', 'severity': 'High', 'progression': 'Defoliation increases as the disease moves upward.', 'store_id': 'mancozeb'},
    'Tomato_Late_blight': {'description': 'Large water-soaked lesions spread quickly on leaves.', 'cause': 'Specific pathogen related to Tomato Late blight', 'pesticide': 'Use systemic plus contact fungicide mix recommended for late blight. Common options include metalaxyl + mancozeb or cymoxanil + mancozeb as per label. Repeat at 5-7 day interval during cool wet periods.', 'organic_alternatives': 'Remove infected leaves immediately to lower inoculum. Improve drainage and reduce standing moisture around plants. Use copper-based organic-compatible sprays where permitted.', 'prevention': 'Stop overhead irrigation and keep field drainage open.', 'spray_time': 'Cool cloudy and wet weather can trigger explosive spread; maintain protective spray before such windows.', 'severity': 'High', 'progression': 'Leaves collapse rapidly and infection can move to stems and fruits or tubers.', 'store_id': 'copper_oxychloride'},
    'Tomato_Leaf_Mold': {'description': 'Pale green to yellow patches appear on the upper leaf surface.', 'cause': 'Specific pathogen related to Tomato Leaf Mold', 'pesticide': 'Use a labeled fungicide such as chlorothalonil or mancozeb with full canopy coverage. Repeat at 5-7 day interval if greenhouse humidity stays high. Target the underside of leaves where mold growth is strongest.', 'organic_alternatives': 'Remove infected leaves and improve ventilation immediately. Use bio-fungicide products only after lowering humidity around the crop. Avoid overcrowded pruning that traps moisture inside the canopy.', 'prevention': 'Water early in the day and avoid overhead irrigation that keeps leaves wet for long hours.', 'spray_time': 'High humidity and poor ventilation favor fast leaf mold development, especially in protected structures.', 'severity': 'High', 'progression': 'Leaves curl, dry and drop when humidity stays high.', 'store_id': 'mancozeb'},
    'Tomato_Septoria_leaf_spot': {'description': 'Many small round spots with grey centers appear on older leaves.', 'cause': 'Specific pathogen related to Tomato Septoria leaf spot', 'pesticide': 'Begin a protectant fungicide schedule with mancozeb or chlorothalonil. Repeat at 5-7 day interval during continued spotting pressure. Cover lower leaves well because infection usually starts from the bottom canopy.', 'organic_alternatives': 'Remove infected lower leaves and keep mulch over exposed soil. Use compost extract or bio-fungicide only after sanitation and pruning. Keep tools clean and avoid working in the field while leaves are wet.', 'prevention': 'Use drip or furrow irrigation to reduce splash from soil to lower leaves.', 'spray_time': 'Rain splash and long periods of leaf wetness will increase new spot formation.', 'severity': 'High', 'progression': 'Heavy spotting leads to fast leaf drop from the lower canopy.', 'store_id': 'mancozeb'},
    'Tomato_Spider_mites_Two_spotted_spider_mite': {'description': 'Tiny yellow specks appear across the leaf surface.', 'cause': 'Specific pathogen related to Tomato Spider mites Two spotted spider mite', 'pesticide': 'Use a registered miticide and rotate active ingredients to avoid resistance. Ensure good spray coverage on the underside of leaves. Repeat only according to label interval after scouting.', 'organic_alternatives': 'Spray neem oil 3-5 ml/L targeting lower leaf surface. Wash hotspots with plain water in the morning where feasible. Conserve predatory mites and beneficial insects.', 'prevention': 'Maintain even moisture; drought stress usually worsens mite outbreaks.', 'spray_time': 'Hot dry weather favors mites, so scout daily in such periods.', 'severity': 'High', 'progression': 'Leaves bronze, dry and curl under heavy infestation.', 'store_id': 'abamectin'},
    'Tomato__Target_Spot': {'description': 'Brown target-like lesions appear with clear circular zoning.', 'cause': 'Specific pathogen related to Tomato  Target Spot', 'pesticide': 'Apply a protectant plus systemic fungicide rotation suitable for target spot. Increase spray coverage in the lower and inner canopy where lesions build up first. Repeat on the labeled interval during warm humid weather.', 'organic_alternatives': 'Remove heavily spotted leaves and improve row ventilation. Use mulch to reduce movement of spores from soil splash. Avoid keeping leaves wet late into the evening.', 'prevention': 'Avoid late-day overhead irrigation and keep leaf wetness periods short.', 'spray_time': 'Warm humid conditions strongly favor new target spot lesions and lesion expansion.', 'severity': 'High', 'progression': 'Leaf drop increases and fruits may also show lesions.', 'store_id': 'mancozeb'},
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {'description': 'Leaves curl upward and become smaller than normal.', 'cause': 'Specific pathogen related to Tomato  Tomato YellowLeaf  Curl Virus', 'pesticide': 'There is no cure for infected leaves, so focus on vector control using recommended whitefly insecticides. Rotate systemic insecticides only according to local label guidance. Remove severely affected plants early to reduce source pressure.', 'organic_alternatives': 'Use yellow sticky traps and reflective mulch to reduce whitefly movement. Spray neem oil 3-5 ml/L to reduce vector feeding pressure. Destroy badly infected plants outside the growing area.', 'prevention': 'Maintain even moisture and avoid plant stress because stressed plants decline faster under virus pressure.', 'spray_time': 'Warm dry weather supports faster whitefly activity, so increase scouting frequency in such periods.', 'severity': 'High', 'progression': 'Plant growth slows sharply and fruit set is reduced.', 'store_id': 'imidacloprid'},
    'Tomato__Tomato_mosaic_virus': {'description': 'Mosaic light and dark green pattern appears on leaves.', 'cause': 'Specific pathogen related to Tomato  Tomato mosaic virus', 'pesticide': 'No curative spray will fix infected leaves, so remove infected plants and focus on sanitation. Disinfect tools, trays, and support wires after working in affected rows. Control sap-sucking pests when needed, based on local scouting.', 'organic_alternatives': 'Wash hands with soap before handling healthy plants after touching infected ones. Use clean seedlings and avoid tobacco contamination near tomato plants. Rogue badly infected plants and dispose of them away from the crop.', 'prevention': 'Maintain steady irrigation to reduce plant stress but avoid moving sap through rough handling during wet operations.', 'spray_time': 'Handling plants in warm dry weather can still spread contamination if tools and hands are not cleaned.', 'severity': 'High', 'progression': 'Plants show stunting and uneven canopy growth.', 'store_id': None},
    'Tomato_healthy': {'description': 'Crop appears healthy.', 'cause': 'N/A', 'pesticide': 'No pesticide required.', 'organic_alternatives': 'N/A', 'prevention': 'Continue regular monitoring.', 'spray_time': 'N/A', 'severity': 'Safe', 'progression': 'Expected to remain healthy.', 'store_id': None},
    "common_rust": {
        "description": "Reddish-brown pustules on leaves.",
        "cause": "Fungal infection (Puccinia sorghi) favored by high humidity and cool temperatures.",
        "pesticide": "Propiconazole 25% EC",
        "organic_alternatives": "Neem oil spray, compost tea, or sulfur-based organic fungicides.",
        "prevention": "Use resistant varieties and ensure good field drainage.",
        "spray_time": "Early morning or late evening.",
        "severity": "Moderate",
        "progression": "Pustules spread to stalks and husks within 7-10 days, reducing yield by up to 20% if untreated.",
        "store_id": "propiconazole",
    },
    "gray_leaf_spot": {
        "description": "Rectangular gray lesions caused by fungus.",
        "cause": "Fungus (Cercospora zeae-maydis) thriving in high humidity and cloudy weather.",
        "pesticide": "Azoxystrobin 23% SC",
        "organic_alternatives": "Bacillus subtilis-based biofungicides.",
        "prevention": "Crop rotation and debris removal.",
        "spray_time": "When weather is dry, preferably morning.",
        "severity": "Moderate",
        "progression": "Lesions will merge and blight entire leaves within a week, severely impacting photosynthesis.",
        "store_id": "azoxystrobin",
    },
    "blight": {
        "description": "Elongated gray lesions.",
        "cause": "Fungal infection (Exserohilum turcicum) often spreading via wind and rain.",
        "pesticide": "Mancozeb 75% WP",
        "organic_alternatives": "Copper-based organic sprays or Neem extract.",
        "prevention": "Avoid overhead irrigation.",
        "spray_time": "Late afternoon to allow overnight drying.",
        "severity": "High",
        "progression": "Rapidly blights upper leaves in 7 days, causing premature death of the plant.",
        "store_id": "mancozeb",
    },
    "healthy": {
        "description": "Crop appears healthy.",
        "cause": "N/A",
        "pesticide": "No pesticide required.",
        "organic_alternatives": "N/A",
        "prevention": "Maintain proper nutrition and regular monitoring.",
        "spray_time": "N/A",
        "severity": "Safe",
        "progression": "Expected to remain healthy with continued good agricultural practices.",
        "store_id": None,
    },
    "normal": {
        "description": "Crop appears healthy.",
        "cause": "N/A",
        "pesticide": "No pesticide required.",
        "organic_alternatives": "N/A",
        "prevention": "Continue regular monitoring.",
        "spray_time": "N/A",
        "severity": "Safe",
        "progression": "Expected to remain healthy.",
        "store_id": None,
    },
    "bacterial_blight": {
        "description": "Angular spots that may merge and dry leaf areas.",
        "cause": "Bacteria (Xanthomonas axonopodis) spreading via rain splash.",
        "pesticide": "Copper oxychloride 50% WP",
        "organic_alternatives": "Baking soda spray or fermented cow dung extract.",
        "prevention": "Use clean seed and avoid overhead irrigation.",
        "spray_time": "Dry, sunny mornings.",
        "severity": "Moderate",
        "progression": "Spots will merge causing leaf necrosis and potential defoliation in 10 days.",
        "store_id": "copper_oxychloride",
    },
    "curl_virus": {
        "description": "Leaf curling and stunted growth due to viral infection.",
        "cause": "Begomovirus transmitted primarily by whiteflies.",
        "pesticide": "Control whitefly vectors as per local advisory.",
        "organic_alternatives": "Yellow sticky traps and Neem oil to deter whiteflies.",
        "prevention": "Use resistant varieties and early vector control.",
        "spray_time": "Evening, when whiteflies are less active.",
        "severity": "High",
        "progression": "Severe stunting and up to 80% yield loss if vectors are not controlled within a week.",
        "store_id": "imidacloprid",
    },
    "fussarium_wilt": {
        "description": "Yellowing and wilting caused by soil-borne fungus.",
        "cause": "Soil-borne fungus (Fusarium oxysporum) attacking roots and vascular system.",
        "pesticide": "Carbendazim 50% WP",
        "organic_alternatives": "Trichoderma viride soil application.",
        "prevention": "Practice crop rotation and ensure good drainage.",
        "spray_time": "Soil drenching in the morning.",
        "severity": "High",
        "progression": "Progressive wilting of entire plant leading to death within 7-10 days.",
        "store_id": "carbendazim",
    },
    "bacterial_leaf_blight": {
        "description": "Water-soaked yellow lesions spreading from leaf tips.",
        "cause": "Bacteria (Xanthomonas oryzae) entering through wounds or natural openings.",
        "pesticide": "Streptocycline and copper spray.",
        "organic_alternatives": "Cow dung slurry spray (20%).",
        "prevention": "Balanced nitrogen and clean irrigation practices.",
        "spray_time": "Late afternoon.",
        "severity": "High",
        "progression": "Lesions will quickly expand down the leaf blade, reducing yield by up to 50% in severe cases over 10 days.",
        "store_id": "copper_oxychloride",
    },
    "bacterial_leaf_streak": {
        "description": "Narrow translucent streaks turning yellow-brown.",
        "cause": "Bacterial infection favored by strong winds and heavy rain.",
        "pesticide": "Copper based bactericide.",
        "organic_alternatives": "Neem seed kernel extract.",
        "prevention": "Use disease-free seeds and proper spacing.",
        "spray_time": "Morning, after dew has dried.",
        "severity": "Moderate",
        "progression": "Streaks will coalesce, turning entire leaves brown and dry within a week.",
        "store_id": "copper_oxychloride",
    },
    "bacterial_panicle_blight": {
        "description": "Discoloration and sterility in panicles.",
        "cause": "Seed-borne bacteria exacerbated by high night temperatures.",
        "pesticide": "Consult local extension guidance for spray timing.",
        "organic_alternatives": "Use of antagonistic bacteria (Pseudomonas fluorescens).",
        "prevention": "Avoid excess nitrogen near flowering stage.",
        "spray_time": "During the heading stage, preferably late evening.",
        "severity": "Moderate",
        "progression": "Can cause up to 40% grain sterility and significant weight loss in infected panicles over 7 days.",
        "store_id": "copper_oxychloride",
    },
    "blast": {
        "description": "Diamond-shaped lesions and possible neck rot.",
        "cause": "Fungus (Magnaporthe oryzae) spreading via airborne spores in high humidity.",
        "pesticide": "Tricyclazole 75% WP",
        "organic_alternatives": "Pseudomonas fluorescens formulation.",
        "prevention": "Use resistant varieties and balanced nutrition.",
        "spray_time": "Early morning to ensure maximum absorption.",
        "severity": "High",
        "progression": "If it reaches the neck, it can cause 'neck blast' leading to complete panicle failure within 10 days.",
        "store_id": "tricyclazole",
    },
    "brown_spot": {
        "description": "Brown circular or oval spots on leaves and grains.",
        "cause": "Fungus (Bipolaris oryzae) prevalent in nutrient-deficient soils.",
        "pesticide": "Mancozeb 75% WP",
        "organic_alternatives": "Improving soil fertility with organic compost.",
        "prevention": "Improve potassium nutrition and field hygiene.",
        "spray_time": "Morning or late afternoon.",
        "severity": "Moderate",
        "progression": "Spots multiply and reduce grain weight and quality within 7-10 days if nutrition isn't corrected.",
        "store_id": "mancozeb",
    },
    "dead_heart": {
        "description": "Central shoot dries and can be pulled out easily.",
        "cause": "Stem borer larvae feeding inside the stem.",
        "pesticide": "Target pest control based on local guidance.",
        "organic_alternatives": "Trichogramma egg parasitoids release.",
        "prevention": "Regular scouting and timely intervention.",
        "spray_time": "Evening, targeting the moth laying stage.",
        "severity": "High",
        "progression": "Tiller will die completely within a few days; pest may move to neighboring tillers.",
        "store_id": None,
    },
    "downy_mildew": {
        "description": "Pale yellow patches with downy growth in humidity.",
        "cause": "Oomycete pathogen favored by high humidity and cool nights.",
        "pesticide": "Metalaxyl plus Mancozeb combination.",
        "organic_alternatives": "Copper fungicides or garlic extract.",
        "prevention": "Improve spacing and reduce leaf wetness period.",
        "spray_time": "Early morning before temperature rises.",
        "severity": "Moderate",
        "progression": "Foliage will become entirely blighted and drop within 7-10 days if humidity persists.",
        "store_id": "mancozeb",
    },
    "hispa": {
        "description": "Scraping damage and white feeding streaks.",
        "cause": "Rice Hispa beetle feeding on the upper epidermis of leaves.",
        "pesticide": "Recommended insecticide per local advisory.",
        "organic_alternatives": "Clipping off and destroying infected leaf tips.",
        "prevention": "Early pest monitoring and field sanitation.",
        "spray_time": "Morning or evening when beetles are active.",
        "severity": "Moderate",
        "progression": "Leaves will wither and die back within a week, severely reducing photosynthetic area.",
        "store_id": None,
    },
    "tungro": {
        "description": "Yellow-orange discoloration and plant stunting.",
        "cause": "Viral disease transmitted by green leafhoppers.",
        "pesticide": "No direct cure; manage leafhopper vectors.",
        "organic_alternatives": "Light traps for catching leafhoppers and Neem oil.",
        "prevention": "Use resistant lines and synchronized planting.",
        "spray_time": "Evening, to target the leafhopper vectors.",
        "severity": "High",
        "progression": "Plant will remain stunted and may not produce any panicles; vectors will spread virus to adjacent fields within days.",
        "store_id": None,
    },
}

# ===============================
# Store Products
# ===============================

store_products = {
    'abamectin': {'name': 'Abamectin 1.9% EC', 'price': 'INR 1500 per liter', 'description': 'Effective miticide and insecticide.'},
    "propiconazole": {
        "name": "Propiconazole 25% EC",
        "price": "INR 850 per liter",
        "description": "Systemic fungicide effective against rust diseases.",
    },
    "azoxystrobin": {
        "name": "Azoxystrobin 23% SC",
        "price": "INR 1200 per liter",
        "description": "Broad spectrum fungicide.",
    },
    "mancozeb": {
        "name": "Mancozeb 75% WP",
        "price": "INR 600 per kg",
        "description": "Protective fungicide for blight control.",
    },
    "tricyclazole": {
        "name": "Tricyclazole 75% WP",
        "price": "INR 950 per kg",
        "description": "Systemic fungicide effective against blast disease in paddy.",
    },
    "carbendazim": {
        "name": "Carbendazim 50% WP",
        "price": "INR 800 per kg",
        "description": "Broad spectrum fungicide for wilt diseases.",
    },
    "copper_oxychloride": {
        "name": "Copper Oxychloride 50% WP",
        "price": "INR 700 per kg",
        "description": "Effective bactericide for leaf blight and bacterial diseases.",
    },
    "imidacloprid": {
        "name": "Imidacloprid 17.8% SL",
        "price": "INR 650 per liter",
        "description": "Systemic insecticide for controlling whiteflies and other vectors.",
    },
}


# ===============================
# Image Preprocessing
# ===============================

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224), color_mode="rgb")
    img_array = image.img_to_array(img).astype("float32")
    return np.expand_dims(img_array, axis=0)


def preprocess_for_crop(img_array, crop_type):
    if not TENSORFLOW_AVAILABLE:
        return img_array / 255.0

    if crop_type == "Cotton":
        return tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array / 255.0


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_leaf_image(filepath):
    img_cv = cv2.imread(filepath)
    if img_cv is None:
        return False
        
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
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
                    break
                    
            if not is_nature_like:
                hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array([10, 20, 20]), np.array([100, 255, 255]))
                ratio = cv2.countNonZero(mask) / (img_cv.shape[0] * img_cv.shape[1])
                if ratio < 0.15:
                    return False
        except Exception as e:
            pass
            
    return True


# ===============================
# Grad-CAM Helpers
# ===============================

def make_gradcam_heatmap(img_array, model, pred_index=None, layer_name=None):
    inner_model = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            inner_model = layer
            break
            
    if inner_model is not None:
        base_model = inner_model
        top_layers = model.layers[model.layers.index(inner_model)+1:]
    else:
        base_model = model
        top_layers = []

    last_conv_layer_name = None
    if layer_name:
        try:
            base_model.get_layer(layer_name)
            last_conv_layer_name = layer_name
        except ValueError:
            pass
            
    if last_conv_layer_name is None:
        for layer in reversed(base_model.layers):
            lname = layer.name.lower()
            if ('conv' in lname or hasattr(layer, 'filters')) and 'bn' not in lname and 'batchnorm' not in lname:
                last_conv_layer_name = layer.name
                break
            
    if last_conv_layer_name is None:
        raise ValueError("Could not find a convolutional layer.")

    grad_model = tf.keras.models.Model(
        inputs=[base_model.inputs],
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        tape.watch(last_conv_layer_output)
        
        for layer in top_layers:
            preds = layer(preds)
            
        if isinstance(preds, list):
            preds = preds[0]
            
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    if grads is None:
        # Fallback if gradient computation fails
        return np.zeros((last_conv_layer_output.shape[1], last_conv_layer_output.shape[2]))
        
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Safe normalization to avoid division by zero
    heatmap = tf.maximum(heatmap, 0)
    max_heat = tf.math.reduce_max(heatmap)
    if max_heat == 0:
        return heatmap.numpy()
        
    heatmap = heatmap / max_heat
    return heatmap.numpy()

def save_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    # Ensure dimensions match
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Apply colormap on scaled uint8 heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Use cv2.addWeighted for correct image blending without overexposure/clipping
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    
    cv2.imwrite(cam_path, superimposed_img)


# ===============================
# Prediction Route
# ===============================

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    advice = None
    image_url = None
    heatmap_url = None
    warning = None
    crop_type = None
    product = None

    if request.method == "POST":
        crop_type = request.form.get("crop")

        if "image" not in request.files:
            warning = "Please upload an image."
            return render_template("index.html", warning=warning, selected_crop=crop_type)

        file = request.files["image"]

        if file.filename == "":
            warning = "No file selected."
            return render_template("index.html", warning=warning, selected_crop=crop_type)

        if crop_type not in {"Corn", "Paddy", "Cotton", "Tomato", "Potato", "Pepper"}:
            warning = "Please select a valid crop type."
            return render_template("index.html", warning=warning, selected_crop=None)

        if not allowed_file(file.filename):
            warning = "Please upload a valid image file (png, jpg, jpeg, webp)."
            return render_template("index.html", warning=warning, selected_crop=crop_type)

        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        safe_name = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(filepath)

        if not is_valid_leaf_image(filepath):
            warning = "Please upload only leaf images. Non-leaf objects detected."
            try:
                os.remove(filepath)
            except OSError:
                pass
            return render_template("index.html", warning=warning, selected_crop=crop_type)

        image_url = url_for("static", filename=f"uploads/{unique_name}")
        img_array = preprocess_for_crop(prepare_image(filepath), crop_type)

        if MODELS_LOADED:
            if crop_type == "Corn":
                model = corn_model
                classes = corn_classes
            elif crop_type == "Paddy":
                model = paddy_model
                classes = paddy_classes
            elif crop_type == "Cotton":
                model = cotton_model
                classes = cotton_classes
            else:
                model = mixed_model
                classes = mixed_classes

            preds = model.predict(img_array, verbose=0)
            
            if model == mixed_model and crop_type in {"Tomato", "Potato", "Pepper"}:
                valid_indices = [i for i, c in enumerate(classes) if c.startswith(crop_type)]
                if valid_indices:
                    class_index = max(valid_indices, key=lambda i: preds[0][i])
                    valid_probs_sum = sum(preds[0][i] for i in valid_indices)
                    if valid_probs_sum > 0:
                        confidence = round((float(preds[0][class_index]) / valid_probs_sum) * 100, 2)
                    else:
                        confidence = round(float(preds[0][class_index]) * 100, 2)
                else:
                    class_index = int(np.argmax(preds[0]))
                    confidence = round(float(np.max(preds[0])) * 100, 2)
            else:
                class_index = int(np.argmax(preds[0]))
                confidence = round(float(np.max(preds[0])) * 100, 2)
                
            prediction = classes[class_index]

            # Generate Grad-CAM Heatmap
            try:
                gradcam_layer = None
                    
                heatmap = make_gradcam_heatmap(img_array, model, class_index, layer_name=gradcam_layer)
                heatmap_filename = f"heatmap_{unique_name}"
                heatmap_path = os.path.join(app.config["UPLOAD_FOLDER"], heatmap_filename)
                save_gradcam(filepath, heatmap, heatmap_path)
                heatmap_url = url_for("static", filename=f"uploads/{heatmap_filename}")
            except Exception as e:
                print(f"Failed to generate Grad-CAM: {e}")
        else:
            # Demo mode - mock predictions
            import random
            if crop_type == "Corn":
                classes = corn_classes
            elif crop_type == "Paddy":
                classes = paddy_classes
            elif crop_type == "Cotton":
                classes = cotton_classes
            else:
                classes = mixed_classes

            prediction = random.choice(classes)
            confidence = round(random.uniform(65, 95), 2)

        if confidence < 60:
            warning = "Low confidence. Please upload a clearer image."

        advice = disease_info.get(
            prediction,
            {
                "description": "No detailed advisory available for this class yet.",
                "cause": "Unknown or undetected pathogen.",
                "pesticide": "Consult a local agricultural expert.",
                "organic_alternatives": "Maintain general soil health.",
                "prevention": "Monitor crop regularly and upload clearer images.",
                "spray_time": "As directed by expert.",
                "severity": "Moderate",
                "progression": "Unknown. Monitor daily for changes.",
                "store_id": None,
            },
        )
        
        product = store_products.get(advice.get("store_id")) if advice and advice.get("store_id") else None

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        advice=advice,
        product=product,
        image_url=image_url,
        heatmap_url=heatmap_url,
        warning=warning,
        selected_crop=crop_type,
    )


# ===============================
# Store Page
# ===============================

@app.route("/store/<product_id>")
def store(product_id):
    product = store_products.get(product_id)
    return render_template("store.html", product=product, product_id=product_id)


# ===============================
# Order Booking Route
# ===============================

@app.route("/process_order/<product_id>", methods=["POST"])
def process_order(product_id):
    product = store_products.get(product_id)
    if not product:
        return "Product not found.", 404
    
    # Get form data
    customer_name = request.form.get("name")
    mobile = request.form.get("mobile")
    address = request.form.get("address")
    city = request.form.get("city")
    pincode = request.form.get("pincode")
    payment_mode = request.form.get("payment_mode")
    
    # Validate form data
    if not all([customer_name, mobile, address, city, pincode, payment_mode]):
        return render_template("store.html", product=product, product_id=product_id, error="Please fill all fields")
    
    # Store order data in session
    order_data = {
        'product_id': product_id,
        'product_name': product['name'],
        'customer_name': customer_name,
        'mobile': mobile,
        'address': address,
        'city': city,
        'pincode': pincode,
        'payment_mode': payment_mode
    }
    session['order_data'] = order_data
    
    if payment_mode == 'cash':
        # Direct success for cash on delivery
        return render_template("order_success.html", product=product, order_data=order_data, payment_mode='cash')
    else:
        # Redirect to payment page for online payment
        return redirect(url_for('payment_page', product_id=product_id))


@app.route("/payment_page/<product_id>")
def payment_page(product_id):
    product = store_products.get(product_id)
    if not product:
        return "Product not found.", 404
    
    order_data = session.get('order_data')
    if not order_data:
        return redirect(url_for('store', product_id=product_id))
    
    return render_template("payment_page.html", product=product, product_id=product_id, order_data=order_data)


@app.route("/payment_success/<product_id>", methods=["GET", "POST"])
def payment_success(product_id):
    product = store_products.get(product_id)
    if not product:
        return "Product not found.", 404
    
    order_data = session.get('order_data')
    if not order_data:
        return redirect(url_for('store', product_id=product_id))
    
    # Process payment (in a real app, verify with payment gateway)
    # For now, we'll just mark it as successful
    
    return render_template("order_success.html", product=product, order_data=order_data, payment_mode='online')


# ===============================
# Run App
# ===============================

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
