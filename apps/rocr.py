import gradio as gr
import torch
import easyocr
import fitz  # PyMuPDF
from tempfile import NamedTemporaryFile
import os
from transformers import BitsAndBytesConfig
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
#from llama_index.core.embeddings import HuggingFaceEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

llama_id = ""
hf_token = ""
os.environ["OPENAI_API_KEY"] = ""

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    llama_id, 
    token=hf_token
    )
stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]
hf_llm = HuggingFaceLLM(
    model_name=llama_id,
    model_kwargs={
    "token": hf_token, 
    # "device_map": "auto",
    "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
    #"quantization_config": quantization_config
    },
    generate_kwargs={
        "do_sample": True,
        "temperature": 0.65,
        "top_p": 0.9,
        "top_k": 50,
    },
    tokenizer_name=llama_id,
    tokenizer_kwargs={"token": hf_token, "max_length":4096},
    stopping_ids=stopping_ids
)
# Use BERT Encoder trained on Romanian language to embed the (OCR-)detected text for retrieval
embeddings = HuggingFaceEmbedding(model_name="dumitrescustefan/bert-base-romanian-cased-v1")

# OCR --------------------------------------------------------------------------------------------

# Initialize EasyOCR reader
reader = easyocr.Reader(['ro'])

# Functions for processing images
def extract_text_from_image(image_bytes):
    with NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        temp_file.write(image_bytes)
        temp_file.seek(0)
        results = reader.readtext(temp_file.name)
    os.remove(temp_file.name)
    return " ".join([result[1] for result in results])

# Function for processing PDF
def process_pdf(file_content):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file_content)
        temp_pdf.seek(0)
        doc = fitz.open(temp_pdf.name)
        all_text = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = []
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                image_info = doc.extract_image(xref)
                img_bytes = image_info["image"]
                with NamedTemporaryFile(suffix=".png", delete=False) as temp_img_file:
                    temp_img_file.write(img_bytes)
                    temp_img_file.flush()
                    temp_img_file.close()
                    ocr_result = reader.readtext(temp_img_file.name)
                    text.extend([line[1] for line in ocr_result])
                    os.unlink(temp_img_file.name)
            all_text.append(" ".join(text))
        doc.close()
    return "\n".join(all_text)

# PROMPTING --------------------------------------------------------------------------------------------

def generate_language_prompt(language):
    prompts = {
        "Romanian": "Summarize the text in Romanian, please! /" + "Rezumatul textului în română!",
        "English": "Summarize the text in English, please!",
        "German": "Summarize the text in German, please! /" + "Zusammenfassung des Textes auf Deutsch!",
        "French": "Summarize the text in French, please! /" + "Résumez le texte en français!",
        "Spanish": "Summarize the text in Spanish, please! /" + "Resumen del texto en español!",
        "Russian": "Summarize the text in Russian, please! /" + "Кратко изложите текст на русском языке!"
    }
    return prompts.get(language, "Summarize the text in English!")  # Default to English if not specified

# GRADIO ----------------------------------------------------------------------------------------------

def process_images(image_files):
    # Extract image text
    all_texts = [extract_text_from_image(image_file) for image_file in image_files]
    combined_text = "\n".join(all_texts)
    return combined_text

def summarize_text(image_files, language):

    # Extract image text using OCR
    all_texts = [extract_text_from_image(image_file) for image_file in image_files]

    # Save all extracted texts into one large .txt file for retrieval mechanism
    with open(os.path.join('temp', 'output_text_file.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_texts))

    # Load the texts as documents
    reader = SimpleDirectoryReader(input_dir='temp', recursive=True)
    documents = reader.load_data(num_workers=1)

    Settings.embed_model = embeddings
    Settings.llm = hf_llm
    # Settings.num_output = 2048
    # Settings.context_window = 3900
    # Settings.chunk_size = 1024
    # Settings.chunk_overlap = 64

    # Create index from extracted texts
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    # Prompting:
    # Generate the language-specific prompt
    init_prompt = """
        You are an expert in summarization of 
        old documents of romanian literature.
        Please act according to the following instructions
        """
    language_prompt = generate_language_prompt(language)
    query = (
         init_prompt +
         language_prompt
        )

    # Query the index with the language-specific prompt
    summary = query_engine.query(query)
    return summary

css = """
    body { background-color: #282B4B !important; }
    .tab-nav { background-color: #FFFFFF; }
    .gradio-app { background-color: #282B4B; }
    .gradio-container { background-color: #282B4B; }
    .tab_content { background-color: #ffffff; border-radius: 12px; padding: 20px; box-shadow: 0 0 8px rgba(0,0,0,0.1); }
    .button, .input_file { background-color: #4CAF50; color: white; }
"""

# Create unified interface with tabs
with gr.Blocks(css=css) as app:
    with gr.Row():
        gr.Image(value="data_img/shift_logo.png", width=50)
    # TAB 1
    with gr.Tab("OCR and Summarization"):
        with gr.Row():
            image_input = gr.Files(label="Upload Image Files", type="binary", file_count='single')
            language = gr.Dropdown(label="Summarization Language", choices=["Romanian", "English", "German", "French", "Spanish", "Russian"], value="English")
            
        image_output = gr.Textbox(label="OCR Detected Text", lines=10)
        summarization_output = gr.Textbox(label="Summarized Text", lines=10)

        process_images_button = gr.Button("Detect Text")
        # Action
        process_images_button.click(
            process_images, 
            inputs=image_input, 
            outputs=image_output
        )
        
        summarize_button = gr.Button("Summarize")
        # Action
        summarize_button.click(
            summarize_text, 
            inputs=[image_input, language], 
            outputs=summarization_output
        )
    # TAB 2
    with gr.Tab("PDF OCR Extraction"):
        pdf_input = gr.File(label="Upload PDF File", type="binary")
        process_pdf_button = gr.Button("Extract Text from PDF")
        pdf_output = gr.Textbox(label="Extracted Text", lines=10)
        # Action
        process_pdf_button.click(
            process_pdf, 
            inputs=pdf_input, 
            outputs=pdf_output
            )

app.launch()
