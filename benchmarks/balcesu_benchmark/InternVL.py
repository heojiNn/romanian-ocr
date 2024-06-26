from PIL import Image
import Levenshtein

# Read the content of the file
with open('ocr/benchmark/InternVL.txt', 'r', encoding='utf-8') as file:
    original_text = file.read()

# Put the wanted output
with open('ocr/benchmark/Balscescu_Groundtruth.txt', 'r', encoding='utf-8') as p:
    predicted_text = p.read()

# Calculate Levenshtein distance
distance = Levenshtein.distance(original_text, predicted_text)
print("InternVL" ,distance) # 180


# Read the content of the file
with open('ocr/benchmark/Mini-InternVL.txt', 'r', encoding='utf-8') as file:
    original_text = file.read()

# Calculate Levenshtein distance
distance = Levenshtein.distance(original_text, predicted_text)
print("Mini-InternVL" ,distance) # 785






