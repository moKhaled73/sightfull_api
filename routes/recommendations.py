from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import io
import os
import sys
import pandas as pd
from PIL import Image
import google.generativeai as genai
import numpy as np
import json


dataset_path = "uicrit/uicrit_public.csv"
images_folder_path = "uicrit/filtered_unique_uis/filtered_unique_uis"
encoded_path = 'uicrit/encoded_images.pkl'
core_path = 'uicrit'

sys.path.append(core_path)

from core.functions import * 
from core.entities import * 
from core.similarity_model import * 

df = pd.read_csv(dataset_path)
data, data_dict = createUICritData(df) 



def extract_json_response(response: str) -> list:
    start_flag = "START_JSON_RESPONSE"
    end_flag = "END_JSON_RESPONSE"

    try:
        # Extract text between flags
        json_text = response.split(start_flag)[1].split(end_flag)[0].strip()

        # Remove ````json` if present
        json_text = json_text.replace("```json", "").replace("```", "").strip()

        # Finding the beginning and end of the JSON
        json_start = min(json_text.find("{"), json_text.find("["))
        json_end = max(json_text.rfind("}"), json_text.rfind("]"))

        # Trim text to get JSON only
        clean_json = json_text[json_start:json_end + 1]

        return clean_json

    except (IndexError, json.JSONDecodeError) as e:
        print("Error extracting JSON:", e)
        return '[]'

def getResponse(input_image, few_shot_images, few_shot_tasks, guidelines):
    # Dynamically generate task descriptions from few_shot_tasks
    tasks_description = ""
    for i, task in enumerate(few_shot_tasks):
        tasks_description += f"tasks for image {i + 1}: {task}. "

    prompt = (
        f"The input image is the first image, followed by few-shot images. "
        f"Analyze the style of the {tasks_description} "
        f"Using this style and following the {guidelines}, generate new recommendations that fit the input image (the first image) in the same format, but are specific to the input image. "
        "For each recommendation, provide an accurate bounding box in the format [x1, y1, x2, y2] that matches the area of the image the recommendation refers to. "
        "Ensure that the bounding box is accurate and aligned with the described issue in the image. "
        "Do not repeat any comment or recommendation that has already been suggested earlier in this response. "
        "Return the response enclosed within special flags: START_JSON_RESPONSE and END_JSON_RESPONSE without any additional text or comments."
    )

    # Generate response using the model
    response = model.generate_content([prompt] + few_shot_images + [ input_image])

    # Extract and print the text part of the response
    response_text = response._result.candidates[0].content.parts[0].text
    return response_text

def get_few_shot(visiual_similarities_df, num_few_shot):
    few_shot_images = []
    for i in range(num_few_shot):
        img = Image.open(os.path.join(images_folder_path, visiual_similarities_df.iloc[i, 0]))
        few_shot_images.append(img)
        
    few_shot_tasks = []
    for i in range(num_few_shot):
        rico_id = visiual_similarities_df.iloc[i, 0].split('.')[0]
        task = json.dumps([t.to_json() for t in data_dict[int(rico_id)].tasks])
        few_shot_tasks.append(task)
    
    few_shot_comments = []  # List to store the comments for each few-shot image
    for i in range(num_few_shot):
        # Get the rico_id from the visual similarities DataFrame
        rico_id = visiual_similarities_df.iloc[i, 0].split('.')[0]
        
        # Get the tasks for the corresponding rico_id from the data_dict
        tasks = data_dict[int(rico_id)].tasks
        
        # Prepare a list to store all comments for the tasks
        comments_dict = []
        
        # Loop through each task to collect its comments
        for task in tasks:
            for comment in task.comments:
                comments_dict.append(comment.to_json())  # Use to_json() method if available
        
        # Encode the entire list of comments to JSON format
        comment_json = json.dumps(comments_dict)
        
        # Append the JSON-encoded comments to the few_shot_comments list
        few_shot_comments.append(comment_json)

    return few_shot_images, few_shot_tasks, few_shot_comments


router = APIRouter()

genai.configure(api_key="AIzaSyBswcFFQUb9j6DL-8YD5yBKkfhBWDktIsQ")
model = genai.GenerativeModel("gemini-1.5-flash")


@router.post("/gemini_recommendations")
async def generate_ui_recommendations(image: UploadFile = File(...), prompt: str = Form(...), coordinations: Optional[str] = Form(None)):
    try:
        image_content = await image.read()

        # Convert binary data to a PIL Image
        image_pil = Image.open(io.BytesIO(image_content))

        if(coordinations) :
            try:
                # Convert comma-separated string to individual values
                coords = coordinations.split(",")
                if len(coords) != 4:
                    raise ValueError("Expected four comma-separated values.")

                x, y, width, height = map(int, coords)

                # Update prompt with bounding box details
                prompt = (f"Provide UI recommendations for the element at "
                          f"position ({x}, {y}) with width {width} and height {height}, "
                          f"based on the following prompt: {prompt}")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid coordinations format. Expected format: 'x,y,width,height'")
        else :
            prompt = "Provide UI recommendations to enhance this design based on the following prompt:" + prompt
        
        response = model.generate_content([prompt, image_pil])
        return {"recommendations": response.text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/our_recommendations")
async def our_recommendations(image: UploadFile = File(...), guideline: str = Form(...)):
    
    image_content = await image.read()

    # Convert binary data to a PIL Image
    image_pil = Image.open(io.BytesIO(image_content))

    # Convert the PIL Image to a NumPy array (buffer array)
    image_array = np.array(image_pil)

    visiual_similarities_df = getSimilarImages(images_folder_path, image_array , encoded_path) 

    few_shot_images, few_shot_tasks, few_shot_comments =  get_few_shot(visiual_similarities_df, 2)

    response = getResponse(image_pil, few_shot_images, few_shot_comments , guideline)
    response_json = json.loads(extract_json_response(response))
    comments_from_json = [Comment.from_json(comment) for comment in response_json] 
    return {"recommendations": comments_from_json}