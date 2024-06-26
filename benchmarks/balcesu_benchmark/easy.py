import easyocr
reader = easyocr.Reader(['ro']) # this needs to run only once to load the model into memory
result = reader.readtext('ocr\Balcescu_Test.jpg')
# print(result)
# Iterate over the result list and extract text predictions
text_predictions = [entry[1] for entry in result]

# Concatenate text predictions into a single string
full_text = ' '.join(text_predictions)

# Write full text to a text file
with open('ocr\easyocr.txt', 'w', encoding='utf-8') as f:
    f.write(full_text)

# Alternatively, you can print the full text
print(full_text)

# Total words: 353
# Total characters: 2030
# Levenshtein Distance: 137