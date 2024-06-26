
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

# load image from the IAM database
url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')

model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
# initialize the encoder from a pretrained ViT and the decoder from a pretrained BERT model. 
# Note that the cross-attention layers will be randomly initialized, and need to be fine-tuned on a downstream dataset
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/vit-base-patch16-224-in21k", 
    "readerbench/RoBERT-large"
)

pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
