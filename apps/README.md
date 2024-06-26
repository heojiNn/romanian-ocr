# OCR and Text Summarization App

This repository provides an OCR and text summarization application built with Python. The application uses EasyOCR for optical character recognition (OCR), PyMuPDF for processing PDF files, and HuggingFace Transformers for text summarization. The user interface is created with Gradio, providing an easy-to-use web interface for uploading images and PDFs, extracting text, and generating summaries in multiple languages.

## Features

- **OCR for Images**: Extracts text from images using EasyOCR.
- **PDF Processing**: Extracts text from PDF files, including text from images within the PDF.
- **Text Summarization**: Summarizes extracted text in various languages using a HuggingFace language model.
- **Multi-language Support**: Summarization available in English, Romanian, German, French, Spanish, and Russian.
- **User-friendly Interface**: Simple web interface built with Gradio, supporting file uploads and text processing with a few clicks.

## Requirements

- Python 3.7+
- set keys in rocr.py
- Install dependencies:

  ```bash
  pip install gradio torch easyocr pymupdf transformers llama-index