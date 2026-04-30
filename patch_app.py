import sys
import ast
sys.path.append('C:\\Users\\acer\\Desktop\\Major_Project\\trail')
from utils import SYMPTOM_LIBRARY, RECOMMENDATION_LIBRARY

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

new_store_products = {}
new_disease_info = {}

for cls in mixed_classes:
    lowered = cls.lower()
    rec_key = "default"
    for key in RECOMMENDATION_LIBRARY.keys():
        if key != "default" and key in lowered:
            rec_key = key
            break
            
    rec = RECOMMENDATION_LIBRARY[rec_key]
    
    symp_key = None
    for key in SYMPTOM_LIBRARY.keys():
        if key in lowered:
            symp_key = key
            break
            
    symptoms = SYMPTOM_LIBRARY[symp_key] if symp_key else ["Visible leaf symptoms are present."]
    
    desc = symptoms[0]
    cause = "Specific pathogen related to " + cls.replace("_", " ")
    progression = "Symptoms may expand over the next 7 days."
    if len(symptoms) > 1: progression = symptoms[-1]
    
    organic = " ".join(rec.get("organic_treatment", ["Maintain field hygiene."]))
    pesticide = " ".join(rec.get("chemical_treatment", ["Consult local expert."]))
    prevention = rec.get("irrigation", "Maintain uniform irrigation.")
    spray_time = rec.get("weather", "Spray during calm weather.")
    
    store_id = None
    if "Copper oxychloride" in rec.get("recommended", []): store_id = "copper_oxychloride"
    elif "Mancozeb" in rec.get("recommended", []): store_id = "mancozeb"
    elif "Chlorothalonil" in rec.get("recommended", []):
        store_id = "chlorothalonil"
        new_store_products["chlorothalonil"] = {"name": "Chlorothalonil 75% WP", "price": "INR 900 per kg", "description": "Broad spectrum protectant fungicide."}
    elif "Imidacloprid" in rec.get("recommended", []): store_id = "imidacloprid"
    elif "Abamectin" in rec.get("recommended", []):
        store_id = "abamectin"
        new_store_products["abamectin"] = {"name": "Abamectin 1.9% EC", "price": "INR 1500 per liter", "description": "Effective miticide and insecticide."}
    elif "Azoxystrobin" in rec.get("recommended", []): store_id = "azoxystrobin"
        
    severity = "High" if "healthy" not in lowered else "Safe"
    if "healthy" in lowered:
        pesticide = "No pesticide required."
        organic = "N/A"
        cause = "N/A"
        progression = "Expected to remain healthy."
        desc = "Crop appears healthy."
        prevention = "Continue regular monitoring."
        spray_time = "N/A"
        store_id = None
        
    new_disease_info[cls] = {
        "description": desc,
        "cause": cause,
        "pesticide": pesticide,
        "organic_alternatives": organic,
        "prevention": prevention,
        "spray_time": spray_time,
        "severity": severity,
        "progression": progression,
        "store_id": store_id
    }

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Very hacky but safe replacement
import json
info_str = ",\n".join(f"    {repr(k)}: {repr(v)}" for k, v in new_disease_info.items())
store_str = ",\n".join(f"    {repr(k)}: {repr(v)}" for k, v in new_store_products.items())

content = content.replace("disease_info = {", "disease_info = {\n" + info_str + ",")
content = content.replace("store_products = {", "store_products = {\n" + store_str + ",")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patched app.py with new disease info and store products.")
