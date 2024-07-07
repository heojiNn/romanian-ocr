# OCR-DocGen

OCR-DocGen is a script designed to generate synthetic text images from a text file for Optical Character Recognition (OCR) training. The script reads a specified number of bytes from an input text file, splits the text into smaller chunks, and creates images for these chunks. These images, along with their corresponding labels, are saved to a specified output directory.

## Usage

### Command-Line Args:

- input_path: Path to the input text file.
- output_dir: Directory to save the output images and labels.
- --bytes_to_read: Number of bytes to read from the input file. Default is 1 GiB (1,073,741,824 bytes).
- --chunk_count: Number of text chunks per image. Default is 50.
- --chunk_length: Maximum length of each text chunk. Default is 100.

### For single-line, max-char=10, image generation, use the following command:

```sh
python data_generator/generator.py --input_path data_generator/data/input/ro-ro-wiki.txt --output_dir data_generator/data/output/ro-l25-c1-10kb-wiki-wb --bytes_to_read 10000 --chunk_count 1 --chunk_length 25
```

### For single-line, max-char=10, english image generation, use the following command:

```sh
python data_generator/generator.py --input_path data_generator/data/input/wiki-ro.txt --output_dir data_generator/data/output/ro-wiki --bytes_to_read 500000 --chunk_count 1 --chunk_length 100
```

python data_generator/generator.py --input_path data_generator/data/input/oscar_ro_train.txt --output_dir data_generator/data/output/ro-oscar --bytes_to_read 150000 --chunk_count 1 --chunk_length 100

### For multi-line (25 lines), max-char=50, image generation use the following command:

```sh
python data_generator/generator.py --input_path data_generator/data/input/wiki-ro.txt --output_dir data_generator/data/output/ro-l50-c40-wiki2-ml-mid --bytes_to_read 100000 --chunk_count 40 --chunk_length 50
```

If you receive AttributeError 'FreeTypeFont', do the following in the environment:
modify trdg.utils.py:
    from PIL import ImageFont

    def get_text_height(font: ImageFont.FreeTypeFont, text: str) -> int:
        return font.getbbox(text)[3]

Output structure (running example):
The structure of data folder as below.
```
data
├── labels.txt
├── labels.csv
└── images
    ├── image0.png
    ├── image1.png
    ├── image2.png
    └── ...
```


## References:

### Image generation:
- https://github.com/Belval/TextRecognitionDataGenerator
### Training / finetuning:
Output format designed for EasyOCR, as explained here:
- https://github.com/clovaai/deep-text-recognition-benchmark

Alternative labeling schemes possible by using: --create_csv and adapting code for respective columns.
### General:
- https://www.freecodecamp.org/news/how-to-fine-tune-easyocr-with-a-synthetic-dataset/
