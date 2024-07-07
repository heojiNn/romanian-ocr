import matplotlib.pyplot as plt
from trdg.generators import GeneratorFromStrings
from PIL import Image, ImageEnhance, ImageFont
import argparse
import os
import csv
from tqdm import tqdm
import time
import random
from fontTools.ttLib import TTFont
import numpy as np


def split_text(text, wordbased=False, min_length=1, max_length=100):
    """
    Splits text into smaller chunks of random length between specified minimum and maximum lengths.
    
    Args:
        text (str): The input text to be split.
        wordbased (bool): Splitting based on words.
        min_length (int): The minimum length of each chunk.
        max_length (int): The maximum length of each chunk.
        
    Returns:
        list: A list of text chunks.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    if not wordbased:
        i = 0
        while i < len(words):
            if random.random() < 0.4857:
                chunk_length = random.choice([1])
                chunk = ' '.join(words[i:i+chunk_length])
                if len(chunk) <= max_length:
                    chunks.append(chunk)
                i += chunk_length
            elif random.random() < 0.1714:
                chunk_length = random.choice([2])
                chunk = ' '.join(words[i:i+chunk_length])
                if len(chunk) <= max_length:
                    chunks.append(chunk)
                i += chunk_length
            elif random.random() < 0.1071:
                chunk_length = random.choice([3])
                chunk = ' '.join(words[i:i+chunk_length])
                if len(chunk) <= max_length:
                    chunks.append(chunk)
                i += chunk_length
            else:
                target_length = int(np.random.beta(2, 3) * (max_length - min_length) + min_length)
                while i < len(words) and current_length + len(words[i]) + 1 <= target_length:
                    current_chunk.append(words[i])
                    current_length += len(words[i]) + 1
                    i += 1
                
                if current_chunk:
                    chunk = ' '.join(current_chunk)
                    if min_length <= len(chunk) and len(chunk) <= max_length:
                        chunks.append(chunk)
                    current_chunk = [] # reset chunk
                    current_length = 0
        if current_chunk:
            chunk = ' '.join(current_chunk)
            if min_length <= len(chunk) and len(chunk) <= max_length:
                chunks.append(chunk)
    else:
        return words

    return chunks

def split_text_distributed(text, wordbased=False):
    """
    Splits text into smaller chunks based on a target word count distribution.
    The distribution comes from easyocr detected word bounding boxes.
    
    Args:
        text (str): The input text to be split.
        wordbased (bool): Splitting based on words.
        
    Returns:
        list: A list of text chunks.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    # Target distribution for word counts
    target_distribution = {
        1: 0.4857,
        2: 0.1714,
        3: 0.1071,
        4: 0.0643,
        5: 0.0714,
        6: 0.0429,
        7: 0.0,
        8: 0.0286,
        9: 0.0071,
        10: 0.0071,
        11: 0.0143
    }

    # Convert the distribution to a list of weights
    weights = [target_distribution.get(i, 0) for i in range(1, max(target_distribution.keys()) + 1)]

    if not wordbased:
        for word in words:
            if not current_chunk:
                # Select a target length based on the target distribution
                target_length = np.random.choice(len(weights), p=weights) + 1

            if current_length + len(word) + 1 > target_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
                target_length = np.random.choice(len(weights), p=weights) + 1
            else:
                current_chunk.append(word)
                current_length += len(word) + 1

        if current_chunk:
            chunks.append(' '.join(current_chunk))
    else:
        return words

    return chunks

def can_render_chars(font_path, characters):
    """
    Check if the font can render the specified characters.
    
    Args:
        font_path (str): The path to the font file.
        characters (str): The characters to check for support.
        
    Returns:
        bool: True if the font can render all specified characters, False otherwise.
        list: A list of characters that the font is missing.
    """
    font = TTFont(font_path)
    cmap = font['cmap'].getBestCmap()
    missing_chars = [char for char in characters if ord(char) not in cmap]
    return (len(missing_chars) == 0), missing_chars

def get_random_font(font_dir, test_characters):
    """
    # TODO: Get only fonts, that are case-sensitive
    Get a random font that supports the specified characters.
    
    Args:
        font_dir (str): The directory containing font files.
        test_characters (str): The characters to test for support.
        
    Returns:
        list: A list of paths to font files that support the specified characters.
    """
    supported_fonts = []
    fonts = [os.path.join(font_dir, font) for font in os.listdir(font_dir) if font.endswith('.ttf')]
    for font_path in fonts:
        try:
            can_render, missing_chars = can_render_chars(font_path, test_characters)
            if can_render:
                supported_fonts.append(font_path)
            else:
                print(f"Font {font_path} is missing characters: {''.join(missing_chars)}")
        except Exception as e:
            print(f"Error with font {font_path}: {e}")
    
    return supported_fonts

def log_config(config_path, params):
    """
    Log the generator parameters to a config file.
    
    Args:
        config_path (str): The path to the config file.
        params (dict): The dictionary of parameters to log.
    """
    with open(config_path, 'a') as f:
        f.write(f"{params}\n")

def generate_image_pairs(input_path, output_dir, bytes_to_read, chunk_count=50,
                         min_chunk_length=1, max_chunk_length=100, create_csv=False,
                         wordbased=False):
    """
    Generates and labels images for OCR detection from a text file.
    
    Args:
        input_path (str): Path to the input text file.
        output_dir (str): Directory to save the output images and labels.
        bytes_to_read (int): Number of bytes to read from the input file.
        chunk_count (int, optional): Number of text chunks per image. Default is 50.
        min_chunk_length (int, optional): Minimum length of each text chunk. Default is 1.
        max_chunk_length (int, optional): Maximum length of each text chunk. Default is 100.
    
    This function reads a specified number of bytes from a text file, splits the text into smaller chunks,
    generates images for each chunk, and combines multiple chunks into a single image. The resulting images
    and their labels are saved to the specified output directory.
    
            # AttributeError 'FreeTypeFont':
            # modify trdg.utils.py:
                from PIL import ImageFont

                def get_text_height(font: ImageFont.FreeTypeFont, text: str) -> int:
                    return font.getbbox(text)[3]

            # for custom backgrounds:
            # in env/trdg/generators/ add images/palette.jpg from assets/palette.jpg
    """
    start_time = time.time()

    with open(input_path, 'r', encoding='utf-8') as file:
        text = file.read(bytes_to_read)

    text_chunks = split_text(text, wordbased=wordbased, 
                             min_length=min_chunk_length, max_length=max_chunk_length)
    os.makedirs(output_dir, exist_ok=True)

    if create_csv:
        csv_file_path = os.path.join(output_dir, 'labels.csv')
        csv_file = open(csv_file_path, mode='w', newline='', encoding='utf-8')
        writer = csv.writer(csv_file)
        writer.writerow(['img_path', 'text'])
    else:
        csv_file = None
        writer = None

    k = 0 # file_index
    images = [] # collect chunk-level images -> combine to total image later.
    page_string = '' # string containing all chunk strings in one page.
    labels_txt_path = os.path.join(output_dir, 'labels.txt') # create .txt as default
    image_dir_path = os.path.join(output_dir, 'images')

    try:
        os.mkdir(image_dir_path)
    except FileExistsError:
        print(f"Directory '{image_dir_path}' already exists")
    except Exception as e:
        print(f"An error occurred: {e}")

    font_dir = 'data_generator/assets/fonts'
    romanian_characters = 'ăâîșțĂÂÎȘȚ'
    supported_fonts  = get_random_font(font_dir, romanian_characters)
    if not supported_fonts:
        print("No fonts support the required Romanian characters.")
        return

    with open(labels_txt_path, "a", encoding='utf-8') as f:
        for i, chunk in enumerate(tqdm(text_chunks, desc="Generating chunks..")):
            font_path = random.choice(supported_fonts)
            generator = GeneratorFromStrings( # play around with data augmentation.
                [chunk],
                # count=-1,
                fonts=[font_path],
                language="latin",
                size=random.randint(48, 128), # 128
                skewing_angle=random.randint(0, 2),  # In degrees counter clockwise.
                random_skew=True,
                blur=random.randint(0, 3),  # [0, 1, 2, 3] # Standard deviation of the Gaussian kernel. Either a sequence of two numbers for x and y, or a single number for both.
                random_blur=True,
                background_type=random.randint(0, 3),  # [0, 1, 2]
                distorsion_type=random.randint(0, 3), # [0, 1, 2, else]
                distorsion_orientation=random.randint(0, 2), # [0, 1, 2]
                # is_handwritten=False,
                # width=-1,
                # alignment=1,
                text_color="#282828",
                orientation=0, # 0 = horizontal text, 1 = vertical text
                space_width=random.uniform(0.5, 1.5),
                character_spacing=random.randint(0, 5),
                margins=(random.randint(5, 15), random.randint(5, 15), random.randint(5, 15), random.randint(5, 15)),
                fit=random.choice([True, False]),
                output_mask=False,
                word_split=False,
                # image_dir=os.path.join(
                #     "..", os.path.split(os.path.realpath(__file__))[0], "images"
                # ),
                stroke_width=random.randint(0, 2), #0
                stroke_fill="#282828",
                image_mode="RGB",
                output_bboxes=0,
                rtl=False,
            )
        
            for img, lbl in generator: 
                if img is not None:  # Ensure img is not None before processing
                    pil_img = ImageEnhance.Sharpness(img).enhance(random.uniform(1.2, 1.8))  # Adjusted sharpness enhancement
                    pil_img = ImageEnhance.Contrast(pil_img).enhance(random.uniform(1.2, 1.8))  # Adjusted contrast enhancement
                    images.append(pil_img)
                    page_string += lbl + ' '
                break

            if ((i + 1) % chunk_count == 0 or i == len(text_chunks) - 1):
                if images:  # Ensure images list is not empty
                    max_width = max(img.width for img in images)
                    line_height = max(img.height for img in images)
                    total_height = len(images) * line_height
                    combined_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))
                    current_height = 0

                    for img in images:
                        combined_image.paste(img, (0, current_height))
                        current_height += line_height

                    img_filename = os.path.join(image_dir_path, f'image{k}.png')
                    combined_image.save(img_filename)

                    f.write(f'image{k}.png\t{page_string}\n')  # .txt
                    if create_csv:
                        writer.writerow([f'image{k}.png', page_string.strip()]) # .csv

                    k += 1
                    images = []
                    page_string = ''

    print(
        'Generation time:', int(time.time() - start_time), 'seconds',
        '\nImage count:', k, 
        # data size
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text images from a text file.")
    parser.add_argument("--input_path", type=str, help="Path to the input text file (can be as large as 60 GB or larger).")
    parser.add_argument("--output_dir", type=str, help="Directory to save the output images and labels.")
    parser.add_argument("--bytes_to_read", type=int, default=10 ** 9, help="Number of bytes to read from the input file. Default 1 GB.")
    parser.add_argument("--chunk_count", type=int, default=50, help="Number of text-rows per page.")
    parser.add_argument("--chunk_length", type=int, default=100, help="Maximum length of text row.")
    parser.add_argument("--create_csv", type=bool, default=True, help="Whether to create a CSV file with labels. Default is True.")
    parser.add_argument("--wordbased", type=bool, default=False, help="Word-per-image (means word per line in multi-line mode)")

    args = parser.parse_args()

    generate_image_pairs(
        input_path=args.input_path, 
        output_dir=args.output_dir, 
        bytes_to_read=args.bytes_to_read, 
        chunk_count=args.chunk_count, 
        min_chunk_length=1,
        max_chunk_length=args.chunk_length,
        create_csv=args.create_csv,
        wordbased=args.wordbased
    )
