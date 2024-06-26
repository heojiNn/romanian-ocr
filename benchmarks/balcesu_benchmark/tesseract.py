import pytesseract
from PIL import Image
import Levenshtein

# Load an image from disk
image = Image.open('ocr/Clujul.jpg')

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Phili\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Use Tesseract to extract text from the image
extracted_text = pytesseract.image_to_string(image)

# Write full text to a text file
with open('ocr/tesseract_clujul.txt', 'w', encoding='utf-8') as f:
    f.write(extracted_text)

with open('ocr/openai-tesseract.txt', 'r', encoding='utf-8') as o:
    original_text = o.read()

# Put the wanted output
with open('ocr/Balscescu_Groundtruth.txt', 'r', encoding='utf-8') as p:
    predicted_text = p.read()

# Calculate Levenshtein distance
distance = Levenshtein.distance(original_text, predicted_text)
print(distance)

# Balcesu Test:
# Levenshtein Distance: 186
# Total characters: 2030

# Clujul Test:
# Levenshtein Distance: 67
# 