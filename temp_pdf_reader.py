import pypdf
import sys

pdf_path = sys.argv[1]
reader = pypdf.PdfReader(pdf_path)
text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(text)
