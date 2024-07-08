import easyocr
from PIL import Image, ImageDraw
from io import BytesIO
import matplotlib.pyplot as plt
import os
import re
import torch

reader = easyocr.Reader(['ro'])

def split_image_by_lines(image_bytes, max_lines_per_part, buffer=5, drawbbox=True):
    
    image = Image.open(BytesIO(image_bytes))
    results = reader.readtext(image_bytes, detail=1)
    bounding_boxes = [result[0] for result in results]

    if drawbbox:
        image_with_boxes = draw_bounding_boxes(image_bytes, bounding_boxes)
        image_with_boxes.show()  # Display the image with bounding boxes

    # Sort bounding boxes by the top-left y-coordinate
    bounding_boxes.sort(key=lambda box: box[0][1])

    # Calculate the average height of the bounding boxes
    box_heights = [box[2][1] - box[0][1] for box in bounding_boxes]
    avg_height = sum(box_heights) / len(box_heights)
    min_height = avg_height * max_lines_per_part

    parts = []
    current_part_boxes = []
    current_lines = 0
    previous_bottom = 0

    for i, box in enumerate(bounding_boxes):
        if current_lines < max_lines_per_part:
            current_part_boxes.append(box)
            current_lines += 1
        else:
            # Get the bounding box coordinates for the current part
            top = max(0, previous_bottom - buffer)
            bottom = max(b[2][1] for b in current_part_boxes)
            height = bottom - top
            if height < min_height:
                bottom = top + min_height  # Ensure part height is greater than min_height
            if bottom <= image.height and top < bottom:  # Ensure valid cropping region
                parts.append(image.crop((0, top, image.width, bottom)))

            # Reset for the next part
            current_part_boxes = [box]
            current_lines = 1
            previous_bottom = bottom

            # If the last part overlaps with the next box, include the overlapping boxes
            while i < len(bounding_boxes) - 1 and bounding_boxes[i + 1][0][1] < bottom:
                #current_part_boxes.append(bounding_boxes[i + 1])
                i += 1

    # Add the last part
    if current_part_boxes:
        top = max(0, previous_bottom - buffer)
        bottom = max(b[2][1] for b in current_part_boxes)
        height = bottom - top
        if height < min_height:
            bottom = top + min_height  # Ensure part height is greater than min_height
        if bottom <= image.height and top < bottom:  # Ensure valid cropping region
            parts.append(image.crop((0, top, image.width, bottom)))

    return parts

def create_images_from_bounding_boxes(image_bytes, buffer=5, drawbbox=True):
    image = Image.open(BytesIO(image_bytes))
    results = reader.readtext(image_bytes, detail=1)

    # Extract bounding boxes of detected text
    bounding_boxes = [result[0] for result in results]

    if drawbbox:
        image_with_boxes = draw_bounding_boxes(image_bytes, bounding_boxes)
        image_with_boxes.show()  # Display the image with bounding boxes

    parts = []

    for box in bounding_boxes:
        # Extract coordinates
        left = min(point[0] for point in box) - buffer
        top = min(point[1] for point in box) - buffer
        right = max(point[0] for point in box) + buffer
        bottom = max(point[1] for point in box) + buffer
        
        # Ensure coordinates are within the image dimensions
        left = max(0, left)
        top = max(0, top)
        right = min(image.width, right)
        bottom = min(image.height, bottom)
        
        # Crop the image
        part = image.crop((left, top, right, bottom))
        parts.append(part)
    
    return parts

def create_images_from_bounding_boxes_wordbased(image_bytes, buffer=5, drawbbox=True):
    image = Image.open(BytesIO(image_bytes))
    results = reader.readtext(image_bytes, detail=1)

    # Extract bounding boxes and detected text
    bounding_boxes = [result[0] for result in results]
    detected_texts = [result[1] for result in results]

    # Split detected texts into individual words and map to bounding boxes
    word_boxes = []
    for box, text in zip(bounding_boxes, detected_texts):
        words = text.split()
        if len(words) > 1:
            # If more than one word is detected in a single box, split the box
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]
            width = (max(x_coords) - min(x_coords)) / len(words)
            
            for i, word in enumerate(words):
                left = min(x_coords) + i * width - buffer
                top = min(y_coords) - buffer
                right = left + width + 2 * buffer
                bottom = max(y_coords) + buffer

                # Ensure coordinates are within the image dimensions
                left = max(0, left)
                top = max(0, top)
                right = min(image.width, right)
                bottom = min(image.height, bottom)

                word_boxes.append(((left, top), (right, bottom)))
        else:
            word_boxes.append(box)
    
    if drawbbox:
        image_with_boxes = draw_bounding_boxes(image_bytes, word_boxes)
        image_with_boxes.show()  # Display the image with bounding boxes

    parts = []

    for box in word_boxes:
        # Extract coordinates
        left = min(point[0] for point in box) - buffer
        top = min(point[1] for point in box) - buffer
        right = max(point[0] for point in box) + buffer
        bottom = max(point[1] for point in box) + buffer
        
        # Ensure coordinates are within the image dimensions
        left = max(0, left)
        top = max(0, top)
        right = min(image.width, right)
        bottom = min(image.height, bottom)
        
        # Crop the image
        part = image.crop((left, top, right, bottom))
        parts.append(part)
    
    return parts

def pil_image_to_bytes(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def extract_text_from_image_parts(image_bytes, max_lines_per_part):
    #image_parts = split_image_by_lines(image_bytes, max_lines_per_part)
    image_parts = create_images_from_bounding_boxes(image_bytes)
    full_text = []
    
    for part in image_parts:
        part_bytes = pil_image_to_bytes(part)
        results = reader.readtext(part_bytes)
        part_text = " ".join([result[1] for result in results])
        full_text.append(part_text)
    
    return " ".join(full_text), image_parts

def extract_text_from_image_parts_with_checkpoint(model, image_bytes):
    image_parts = create_images_from_bounding_boxes(image_bytes)
    full_text = []
    
    for part in image_parts:
        part_bytes = pil_image_to_bytes(part)
        results = model(part_bytes)
        part_text = " ".join([result[1] for result in results])
        full_text.append(part_text)
    
    return " ".join(full_text), image_parts

def extract_text(image_bytes):
    results = reader.readtext(image_bytes)
    full_text = [result[1] for result in results]
    return " ".join(full_text)

def draw_bounding_boxes(image_bytes, bounding_boxes):
    image = Image.open(BytesIO(image_bytes))
    draw = ImageDraw.Draw(image)
    
    for box in bounding_boxes:
        # Convert box coordinates to (left, upper, right, lower) format
        left = min([point[0] for point in box])
        upper = min([point[1] for point in box])
        right = max([point[0] for point in box])
        lower = max([point[1] for point in box])
        draw.rectangle([left, upper, right, lower], outline="red", width=2)
    
    return image

def clean_whitespaces(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()  # Remove leading and trailing spaces

######################## LOAD TEXT ########################

# Load image bytes from a local file path
def load_image_bytes(image_path):
    with open(image_path, 'rb') as image_file:
        return image_file.read()

# Example usage
image_path = 'benchmarks/balcesu_benchmark/Balcescu.jpg'  # Update this with the path to your image
image_bytes = load_image_bytes(image_path)

######################## OCR ########################

MAX_LINES = 5
extracted_text, image_parts = extract_text_from_image_parts(image_bytes, max_lines_per_part=MAX_LINES) # extract by parts
extracted_text = clean_whitespaces(extract_text(image_bytes)) # extract total
 
#model = torch.load("easyocr/ro-wiki-singleline25-107/best_accuracy.pth")
#model.eval()

######################## VISUALIZE ########################

# def visualize_image_parts_lines(image_parts):
#     plt.figure(figsize=(10, 20))
#     for i, part in enumerate(image_parts):
#         plt.subplot(len(image_parts), 1, i+1)
#         plt.imshow(part)
#         plt.axis('off')
#     plt.show()

# visualize_image_parts_lines(image_parts[:4])

# import math
# def visualize_image_parts_boxes(image_parts):
#     # Calculate grid size
#     num_parts = len(image_parts)
#     grid_size = math.ceil(math.sqrt(num_parts))
    
#     # Create a plot with a grid of images
#     fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
#     axes = axes.flatten()

#     for ax, part in zip(axes, image_parts):
#         ax.imshow(part)
#         ax.axis('off')

#     # Hide any remaining empty subplots
#     for ax in axes[len(image_parts):]:
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()

# visualize_image_parts_boxes(image_parts)

######################## SAVE IMAGE PARTS ########################

def save_image_parts(image_parts, base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for i, part in enumerate(image_parts):
        part.save(os.path.join(base_path, f'image{i}.png'))

        # f.write(f'image{k}.png\t{label}\n')  # .txt

# Save image parts
save_image_parts(image_parts, 'data_generator/data/balcesu_wb')


######################## SAVE TEXT: #######################

# # Function to find the next available file index
# def get_next_file_index(base_path, base_name):
#     i = 1
#     while os.path.exists(f"{base_path}/{base_name}_{i}.txt"):
#         i += 1
#     return i

# # Define the base path and base name
# base_path = 'ocr/balcesu_benchmark'
# #base_name = f'easyocr_ml{MAX_LINES}'
# base_name = f'easyocr'

# # Get the next available file index
# next_index = get_next_file_index(base_path, base_name)

# # Construct the file name with the next available index
# file_name = f"{base_path}/{base_name}_{next_index}.txt"

# # Write full text to the next available text file
# with open(file_name, 'w', encoding='utf-8') as f:
#     f.write(extracted_text)

################################################

