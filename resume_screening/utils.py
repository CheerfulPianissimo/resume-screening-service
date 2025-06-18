import io
from PyPDF2 import PdfReader

def extract_text(pdf_bytes: bytes) -> str:
    """
    Extracts and returns text content from PDF bytes.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text
