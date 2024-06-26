import matplotlib.pyplot as plt
from trdg.generators import GeneratorFromStrings
from PIL import Image, ImageEnhance
import argparse
import os
import csv
from tqdm import tqdm
import time

def split_text(text, max_length=100):
    """
    Splits text into smaller chunks of specified maximum length.
    
    Args:
        text (str): The input text to be split.
        max_length (int): The maximum length of each chunk.
        
    Returns:
        list: A list of text chunks.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def generate_image_pairs(input_path, output_dir, bytes_to_read, chunk_count=50, chunk_length=100, create_csv=False):
    """
    Generates and labels images for OCR detection from a text file.
    
    Args:
        input_path (str): Path to the input text file.
        output_dir (str): Directory to save the output images and labels.
        bytes_to_read (int): Number of bytes to read from the input file.
        chunk_count (int, optional): Number of text chunks per image. Default is 50.
        chunk_length (int, optional): Maximum length of each text chunk. Default is 100.
    
    This function reads a specified number of bytes from a text file, splits the text into smaller chunks,
    generates images for each chunk, and combines multiple chunks into a single image. The resulting images
    and their labels are saved to the specified output directory.
    """
    start_time = time.time()

    with open(input_path, 'r', encoding='utf-8') as file:
        text = file.read(bytes_to_read)

    text_chunks = split_text(text, max_length=chunk_length)
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

    with open(labels_txt_path, "a", encoding='utf-8') as f:
        for i, chunk in enumerate(tqdm(text_chunks, desc="Generating chunks..")):
            generator = GeneratorFromStrings( # play around with data augmentation.
                [chunk],
                language="latin",
                # size=300,
                # skewing_angle=5,
                # random_skew=False,
                # blur=1,
                # random_blur=False,
                background_type=3,
                #is_handwritten=True
            )

            for img, lbl in generator:
                pil_img = ImageEnhance.Sharpness(img).enhance(2.0)
                pil_img = ImageEnhance.Contrast(pil_img).enhance(2.0)
                pil_img = pil_img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
                images.append(pil_img)
                page_string += lbl + ' '
                break

            if (i % chunk_count == 0) and (i != 0):
                max_width = max(img.width for img in images)
                line_height = max(img.height for img in images)
                resized_images = [img.resize((max_width, line_height), Image.LANCZOS) for img in images]
                total_height = len(resized_images) * line_height
                
                combined_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))
                current_height = 0

                for img in resized_images:
                    combined_image.paste(img, (0, current_height))
                    current_height += line_height

                img_filename = os.path.join(image_dir_path, f'image{k}.png')
                combined_image.save(img_filename)

                f.write(f'image{k}.png\t{page_string}\n')  # .txt
                if create_csv:
                    writer.writerow([img_filename, page_string.strip()]) # .csv

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
    parser.add_argument("input_path", type=str, help="Path to the input text file (can be as large as 60 GB or larger).")
    parser.add_argument("output_dir", type=str, help="Directory to save the output images and labels.")
    parser.add_argument("--bytes_to_read", type=int, default=10 ** 9, help="Number of bytes to read from the input file. Default 1 GB.")
    parser.add_argument("--chunk_count", type=int, default=50, help="Number of text-rows per page.")
    parser.add_argument("--chunk_length", type=int, default=100, help="Maximum length of text row.")
    parser.add_argument("--create_csv", type=bool, default=False, help="Whether to create a CSV file with labels. Default is True.")

    args = parser.parse_args()

    generate_image_pairs(args.input_path, args.output_dir, args.bytes_to_read, args.chunk_count, args.chunk_length)