import requests
import torch
from PIL import Image

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda:0"

# Note that passing the image urls (instead of the actual pil images) to the processor is also possible
# image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")

image = Image.open('ocr/Balcescu_Test.jpg')

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
).to(DEVICE)

# Create inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Please perform OCR on the image +\
             i.e. detect the words written in the romanian language in the image."},
        ]
    },
    # { few-shot example:
    #     "role": "assistant",
    #     "content": [
    #         {"type": "text", "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty."},
    #     ]
    # },
    # {
    #     "role": "user",
    #     "content": [
    #         {"type": "image"},
    #         {"type": "text", "text": "And how about this image?"},
    #     ]
    # },
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}


# Generate
generated_ids = model.generate(**inputs, max_new_tokens=2100)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

# Write full text to a text file
with open('ocr/idefics2.txt', 'w', encoding='utf-8') as f:
    f.write(generated_texts)

print(generated_texts)

# Total words: 353
# Total characters: 2030
# Levenshtein Distance: 1193
