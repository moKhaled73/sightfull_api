from core.entities import UICritData
import cv2
import matplotlib.pyplot as plt

def createUICritData(df):
    # Creating the structured data
    data_dict = {}
    for index, row in df.iterrows():
        rico_id = row['rico_id']
        task = row['task']
        comments = row['comments']
        comments_source = row['comments_source']
        
        if comments == '[]' or comments_source == '[]':
            continue

        # If the rico_id is already in the data_dict, append the new task
        if rico_id not in data_dict:
            data_dict[rico_id] = UICritData(rico_id)
        
        data_dict[rico_id].add_task(task, comments_source, comments)

    # Converting the dictionary to a list for further usage
    data = list(data_dict.values())
    return data , data_dict


# Helper function to display an image with a bounding box
def display_image_with_bounding_box(image_path, bbox):
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Create a copy of the image
    image_copy = image.copy()

    # Get the bounding box and scale it to the image size
    if bbox:
        left = int(bbox[0] * width)
        top = int(bbox[1] * height)
        right = int(bbox[2] * width)
        bottom = int(bbox[3] * height)

        # Draw the bounding box (red color with increased thickness)
        cv2.rectangle(image_copy, (left, top), (right, bottom), (0, 0, 255), 4)  # Red color, thickness 4

    # Convert the image from BGR to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

    # Display the image in a figure
    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide the axes

    # Show the plot
    plt.show()