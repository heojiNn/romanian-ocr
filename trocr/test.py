import pandas as pd
import torch
from transformers import TrOCRProcessor
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
import torch
from datasets import load_metric
from tqdm.notebook import tqdm

from dataset import IAMDataset


df = pd.read_fwf('/content/drive/MyDrive/TrOCR/Tutorial notebooks/IAM/gt_test.txt', header=None)
df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
del df[2]
df.head()
     
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
test_dataset = IAMDataset(root_dir='/content/drive/MyDrive/TrOCR/Tutorial notebooks/IAM/image/',
                           df=df,
                           processor=processor)


test_dataloader = DataLoader(test_dataset, batch_size=8)
batch = next(iter(test_dataloader))

for k,v in batch.items():
  print(k, v.shape)



processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

labels = batch["labels"]
labels[labels == -100] = processor.tokenizer.pad_token_id
label_str = processor.batch_decode(labels, skip_special_tokens=True)
label_str



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/vit-base-patch16-224-in21k", 
    "readerbench/RoBERT-large"
)
model.to(device)



cer = load_metric("cer")



print("Running evaluation...")

for batch in tqdm(test_dataloader):
    # predict using generate
    pixel_values = batch["pixel_values"].to(device)
    outputs = model.generate(pixel_values)

    # decode
    pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
    labels = batch["labels"]
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels, skip_special_tokens=True)

    # add batch to metric
    cer.add_batch(predictions=pred_str, references=label_str)

final_score = cer.compute()

print("Character error rate on test set:", final_score)
