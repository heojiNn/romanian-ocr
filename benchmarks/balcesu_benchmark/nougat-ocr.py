import warnings
import nougat# import NougatOCR

# Filter out specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="nougat.model")

# Initialize NougatOCR
nougat_ocr = NougatOCR(model="0.1.0-base")

# Process a PDF
pdf_path = "C:/Software/Python/Artproject/vlproj/ocr/Balscesu_1page.pdf"
output_dir = "ocr"
nougat_ocr.process_pdf(pdf_path, output_dir)
