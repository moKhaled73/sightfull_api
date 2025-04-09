import pandas as pd
import os
import torch
import open_clip
import pickle
import cv2
from sentence_transformers import util
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)


# Preprocess the images and encode using CLIP model
def imageEncoder(img):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1

# دالة لحساب درجة التشابه بين صورتين
def generateScore(image1, image2):    
    cos_scores = util.pytorch_cos_sim(image1, image2)
    score = round(float(cos_scores[0][0]) * 100, 2)  # تحويل التشابه إلى نسبة مئوية
    return score


def getSimilarImages(images_folder, input_image, encoded_path):
    # Load the pre-encoded images
    with open(encoded_path, "rb") as f:
        encoded_images = pickle.load(f)

    if input_image is None:
        raise ValueError(f"Could not open or find the image: {input_image}")

    # Encode the input image
    encoded_input_img = imageEncoder(input_image).cpu().detach()

    # Compute similarities using pre-encoded images
    similarities = []
    for image_file, encoded_img in encoded_images.items():
    
        # Calculate similarity
        similarity_score = generateScore(encoded_input_img, encoded_img)
        similarities.append({"image_name": image_file, "similarity": similarity_score})

    # Convert to DataFrame
    similarities_df = pd.DataFrame(similarities)
    similarities_df.sort_values(by="similarity", ascending=False, inplace=True, ignore_index=True)
    return similarities_df
