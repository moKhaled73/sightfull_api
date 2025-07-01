from fastapi import APIRouter, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import io

def preprocess_image(img, target_size=(224, 224)):
    # Convert PIL image to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize, convert to array, normalize, and expand dims
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

scores_weights = "models/scores.h5"
scores_model = tf.keras.models.load_model(scores_weights)


router = APIRouter()

@router.post("/")
async def predict_scores(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))  # PIL image

        img_array = preprocess_image(img)  # Use shared preprocessing

        scaler = joblib.load("models/label_scaler.pkl")
        scores = scores_model.predict(img_array)
        original_scores = scaler.inverse_transform(scores)

        scores = original_scores[0].tolist()

        return {
            "visual_aesthetics_score" : float(scores[0]),
            "clarity_readability_score": float(scores[1]),
            "information_hierarchy_score": float(scores[2]),
            "accessibility_score": float(scores[3]),
            "visual_cluttering_score": float(scores[4])
        }    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
