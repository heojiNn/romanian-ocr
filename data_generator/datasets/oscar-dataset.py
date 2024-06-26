from datasets import load_dataset
from tqdm import tqdm

# Load the Romanian subset of the OSCAR corpus
dataset = load_dataset("oscar", "unshuffled_deduplicated_ro")

# Print the dataset details
print(dataset)

# Access the train split
train_split = dataset['train']

# Define the output file path
output_file = "oscar_romanian.txt"

# Write the dataset to a .txt file
with open(output_file, 'w', encoding='utf-8') as f:
    for example in tqdm(train_split):
        f.write(example['text'] + "\n")

print(f"Dataset saved to {output_file}")

# Print the first example
print("First example:")
print(train_split[0])

# Print the first 10 examples
print("\nFirst 10 examples:")
for i in range(10):
    print(f"Example {i + 1}:")
    print(train_split[i]['text'])
    print()




