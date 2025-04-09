from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import cv2
import numpy as np
import tensorflow as tf


def padding(img, shape_r, shape_c, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

def preprocess_image(image) :
    img = np.array(image)
    preprocessed_img = np.zeros((1, 256, 256, 3))
    padded_image = padding(img, 256, 256, 3)
    preprocessed_img[0] = padded_image

    preprocessed_img[:, :, :, 0] -= 103.939
    preprocessed_img[:, :, :, 1] -= 116.779
    preprocessed_img[:, :, :, 2] -= 123.68

    return preprocessed_img

def resize_image_with_aspect_ratio(image, target_size=512):
    # Get the current dimensions of the image
    height, width = image.shape[:2]
    
    # Check if width is greater than height
    if width > height:
        # Calculate the new height keeping the aspect ratio
        new_width = target_size
        new_height = int((target_size / width) * height)
    else:
        # Calculate the new width keeping the aspect ratio
        new_height = target_size
        new_width = int((target_size / height) * width)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_image

def apply_colormap_to_heatmap(heatmap):
    heatmap = np.maximum(heatmap, 0)  # Ensure all values are positive
    heatmap /= heatmap.max()  # Normalize between 0 and 1
    heatmap = np.uint8(255 * heatmap)  # Scale to [0, 255]
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

def crop_image_to_dimension(image, target_value, dim='width'):
    height, width = image.shape[:2]
    
    if dim == 'width':
        # If the target value is greater than the current width, no cropping is done
        if target_value >= width:
            return image
        
        # Calculate how much to crop from each side
        crop_amount = (width - target_value) // 2
        
        # Crop the image by keeping the center and removing equally from left and right
        cropped_image = image[:, crop_amount:width-crop_amount]
    
    elif dim == 'height':
        # If the target value is greater than the current height, no cropping is done
        if target_value >= height:
            return image
        
        # Calculate how much to crop from the top and bottom
        crop_amount = (height - target_value) // 2
        
        # Crop the image by keeping the center and removing equally from top and bottom
        cropped_image = image[crop_amount:height-crop_amount, :]
    
    return cropped_image

def resize_heatmap_to_original_image(heatmap, original_image):
    img = np.array(original_image)
    img = resize_image_with_aspect_ratio(img)
    height, width,c = img.shape
    if(width > height):
        return crop_image_to_dimension(heatmap, height, "height")
    elif(height > width):
        return crop_image_to_dimension(heatmap, width, "width")


heatmap3s_weights = 'models/umsi-3s.h5'
heatmap7s_weights = 'models/umsi-7s.h5'
heatmap3s_model = tf.keras.models.load_model(heatmap3s_weights)
heatmap7s_model = tf.keras.models.load_model(heatmap7s_weights)



router = APIRouter()


# heatmap 3s route
@router.post("/heatmap3s")
async def upload_image(file: UploadFile = File(...)):
    try:    
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # image preprocessing
        preprocessed_img = preprocess_image(image)

        # Step 4: Predict using the model 3s
        predicited_heatmap = heatmap3s_model.predict(preprocessed_img)[0][0]

        # Step 4: Resize the preds_map to match the original image size
        resized_heatmap = resize_heatmap_to_original_image(predicited_heatmap, image)

        # Step 5: Normalize and apply colormap to the heatmap
        colored_heatmap = apply_colormap_to_heatmap(resized_heatmap)

        
        # Step 6: Convert the result to bytes and return it
        _, img_encoded = cv2.imencode('.png', colored_heatmap)
        img_bytes = io.BytesIO(img_encoded.tobytes())
        
        return StreamingResponse(img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# heatmap 7s route
@router.post("/heatmap7s")
async def upload_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # image preprocessing
        preprocessed_img = preprocess_image(image)

        # Step 4: Predict using the model 3s
        predicited_heatmap = heatmap7s_model.predict(preprocessed_img)[0][0]

        # Step 4: Resize the preds_map to match the original image size
        resized_heatmap = resize_heatmap_to_original_image(predicited_heatmap, image)

        # Step 5: Normalize and apply colormap to the heatmap
        colored_heatmap = apply_colormap_to_heatmap(resized_heatmap)

        
        # Step 6: Convert the result to bytes and return it
        _, img_encoded = cv2.imencode('.png', colored_heatmap)
        img_bytes = io.BytesIO(img_encoded.tobytes())
        
        return StreamingResponse(img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
