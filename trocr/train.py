
#!pip install -q transformers
#!pip install -q datasets jiwer


import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import IAMDataset
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel
import torch
from transformers import AdamW
from tqdm.notebook import tqdm


from datasets import load_metric


df = pd.read_fwf('/content/drive/MyDrive/TrOCR/Tutorial notebooks/IAM/gt_test.txt', header=None)
df.rename(columns={0: "img_path", 1: "text"}, inplace=True)
del df[2]
# some file names end with jp instead of jpg, let's fix this
df['img_path'] = df['img_path'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
#df.head()
train_df, test_df = train_test_split(df, test_size=0.2)
# we reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)







processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')

print(processor.tokenizer.pad_token)
print(processor.tokenizer.pad_token_id)
print(processor.tokenizer.eos_token)
print(processor.tokenizer.eos_token_id)
print(processor.tokenizer.unk_token)

train_dataset = IAMDataset(root_dir='/content/drive/MyDrive/TrOCR/Tutorial notebooks/IAM/image/',
                           df=train_df,
                           processor=processor)
eval_dataset = IAMDataset(root_dir='/content/drive/MyDrive/TrOCR/Tutorial notebooks/IAM/image/',
                           df=test_df,
                           processor=processor)


print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))


encoding = train_dataset[0]
for k,v in encoding.items():
  print(k, v.shape)
     
image = Image.open(train_dataset.root_dir + train_df['file_name'][0]).convert("RGB")
labels = encoding['labels']
labels[labels == -100] = processor.tokenizer.pad_token_id
label_str = processor.decode(labels, skip_special_tokens=True)
print(label_str)



train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/vit-base-patch16-224-in21k", 
    "readerbench/RoBERT-large"
)
model.to(device)

# Importantly, we need to set a couple of attributes, namely:

# the attributes required for creating the decoder_input_ids from the labels 
# (the model will automatically create the decoder_input_ids by shifting the labels one
# position to the right and prepending the decoder_start_token_id, as well as replacing
# ids which are -100 by the pad_token_id) 

# the vocabulary size of the model 
# (for the language modeling head on top of the decoder)
# beam-search related parameters which are used when generating text.

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4



cer_metric = load_metric("cer")

def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer
     

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(10):  # loop over the dataset multiple times
   # train
   model.train()
   train_loss = 0.0
   for batch in tqdm(train_dataloader):
      # get the inputs
      for k,v in batch.items():
        batch[k] = v.to(device)

      # forward + backward + optimize
      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      train_loss += loss.item()

   print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
    
   # evaluate
   model.eval()
   valid_cer = 0.0
   with torch.no_grad():
     for batch in tqdm(eval_dataloader):
       # run batch generation
       outputs = model.generate(batch["pixel_values"].to(device))
       # compute metrics
       cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
       valid_cer += cer 

   print("Validation CER:", valid_cer / len(eval_dataloader))

model.save_pretrained(".")
#.from_pretrained(output_dir)

