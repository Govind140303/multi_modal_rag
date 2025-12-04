# ingest_simple.py
import pdfplumber, json, os

INPUT_FILE = "data/qatar_test_doc.pdf"   # put a PDF into data/ and name it sample.pdf (or change this)
OUTPUT_FILE = "data_out/ingested.json"

os.makedirs("data_out", exist_ok=True)

def extract_text_from_pdf(path):
    pages=[]
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": i, "text": text.strip()})
    return pages

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print("ERROR: place a PDF at", INPUT_FILE)
    else:
        pages = extract_text_from_pdf(INPUT_FILE)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)
        print("✅ Ingestion complete — saved to", OUTPUT_FILE)
