import google.generativeai as genai

# Initialize the model
genai.configure(api_key="AIzaSyBOGtNJ3xJvTo_bD7FVw4PDiGmi8QbzMkc")
model = genai.GenerativeModel("gemini-1.5-flash")


def getResponse(input_image, few_shot_images, few_shot_tasks, guidelines):
    # Dynamically generate task descriptions from few_shot_tasks
    tasks_description = ""
    for i, task in enumerate(few_shot_tasks):
        tasks_description += f"tasks for image {i + 1}: {task}. "

    # Refined prompt to emphasize the accuracy of bounding boxes
    prompt = (
        f"Analyze the style of the {tasks_description}"
        f"Using this style and following the {guidelines}, generate new recommendations that fit image three in the same format but are specific to the third image. "
        "For each recommendation, provide an accurate bounding box in the format [x1, y1, x2, y2] that matches the area of the image the recommendation refers to. "
        "Ensure that the bounding box is accurate and aligned with the described issue in the image. "
        "Return the response as plain text without any formatting like JSON, code blocks, or other structured data formats."
    )

    # Generate response using the model
    response = model.generate_content([prompt] + few_shot_images + [input_image])

    # Extract and print the text part of the response
    response_text = response._result.candidates[0].content.parts[0].text
    return response_text