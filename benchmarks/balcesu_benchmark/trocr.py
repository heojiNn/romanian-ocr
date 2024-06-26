from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image

processor = TrOCRProcessor.from_pretrained('models/trocr')
model = VisionEncoderDecoderModel.from_pretrained("models/trocr")

# # load image from the IAM dataset
# url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# Load an image from disk
image = Image.open('ocr/Balcescu_Test.jpg')

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)#[0]

# # Write full text to a text file
# with open('ocr/trocr.txt', 'w', encoding='utf-8') as f:
#     f.write(generated_text)

print(generated_text)
