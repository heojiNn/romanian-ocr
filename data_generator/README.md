# OCR-DocGen

OCR-DocGen is a script designed to generate synthetic text images from a text file for Optical Character Recognition (OCR) training. The script reads a specified number of bytes from an input text file, splits the text into smaller chunks, and creates images for these chunks. These images, along with their corresponding labels, are saved to a specified output directory.

## Usage Example

To run the script, use the following command:

```sh
python generator.py data/ro.txt data/ro-CC-100-ocr --bytes_to_read 1000000 --chunk_count 50 --chunk_length 100
```
Reads 10^{6} characters = 1 MB of text.\
Creates 10^{6} / (chunk_count * chunk_length) images.
Default: ~ 2 * 10^{3} images created. time: 10 minutes.

- input_path: Path to the input text file.
- output_dir: Directory to save the output images and labels.
- --bytes_to_read: Number of bytes to read from the input file. Default is 1 GiB (1,073,741,824 bytes).
- --chunk_count: Number of text chunks per image. Default is 50.
- --chunk_length: Maximum length of each text chunk. Default is 100.

Output structure (running example):
The structure of data folder as below.
```
data
├── labels.txt
└── images
    ├── word_1.png
    ├── word_2.png
    ├── word_3.png
    └── ...
```

```sh
data/ro_default/image0.png
...
data/ro_default/image203.png # not exactly 200 images, since some characters may take more or less space.
data/ro_default/labels.csv
data/ro_default/labels.txt
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
