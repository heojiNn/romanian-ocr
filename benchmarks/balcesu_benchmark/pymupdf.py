import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os

# Configure pytesseract to know the path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Phili\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Function to extract text from pdf
def extract_text_from_pdf(pdf_path, page_number):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number - 1)  # Load the specific page (page_number starts from 1)
        
        # Extract text
        text += page.get_text()

        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            image_text = pytesseract.image_to_string(image)
            text += image_text
        
        doc.close()  # Close the PDF document

    except Exception as e:
        print("An error occurred:", e)

    return text

# Function to save text to a .txt file
def save_text_to_txt(text, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if it doesn't exist
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)
        print("Text saved successfully to:", file_path)
    except Exception as e:
        print("An error occurred while saving text:", e)

# Example usage for extracting text from page 18 of the PDF
pdf_path = 'ocr/Balscesu_Test.pdf'
page_number = 18
extracted_text = extract_text_from_pdf(pdf_path, page_number)

# Save the extracted text to a .txt file
output_file_path = 'ocr/pymupdf.txt'
save_text_to_txt(extracted_text, output_file_path)

# Total words: 353
# Total characters: 2030
# Levenshtein Distance: 2094