"""
ingest.py
---------
One-time ingestion script for the University Knowledge Base RAG app.

What it does:
1. Reads the six university PDFs from the local ./knowledge_base/ folder
   (shipped with the project — no external download step, no API keys,
   no network dependency for this step).
2. Extracts text AND tables from each PDF (tables are converted to Markdown
   so numeric/tabular data — fees, grades, dates — survives cleanly).
3. Splits content into section-aware chunks using the documents' own
   numbered headings (e.g. "1. Welcome", "2.1 Attendance Policy") instead of
   blind fixed-size splitting, so a chunk doesn't cut a policy clause in half.
4. Embeds every chunk locally using a HuggingFace sentence-transformer
   (no API key required for this step).
5. Builds a FAISS index and saves it to ./faiss_index/
   (creates index.faiss + index.pkl, exactly what app.py will load).

Run this once, or whenever a PDF in knowledge_base/ changes:
    python ingest.py
"""

import os
import re

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
KB_DIR = "knowledge_base"      # local folder containing the 6 source PDFs
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Matches lines like "1. Welcome to ANUST" or "4.2 Hostel and Accommodation"
HEADING_PATTERN = re.compile(r"^\s*(\d+(\.\d+)*)\.?\s+[A-Z].{3,80}$")


def load_pdf_paths():
    """Collect every PDF sitting in the local knowledge_base/ folder."""
    if not os.path.isdir(KB_DIR):
        raise SystemExit(f"Folder ./{KB_DIR}/ not found. Create it and place the 6 PDFs inside.")

    pdfs = sorted(f for f in os.listdir(KB_DIR) if f.lower().endswith(".pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in ./{KB_DIR}/. Add the 6 source PDFs there and re-run.")

    print(f"Found {len(pdfs)} PDF(s) in ./{KB_DIR}/: {pdfs}")
    return [os.path.join(KB_DIR, f) for f in pdfs]


def table_to_markdown(table):
    """Convert a pdfplumber-extracted table (list of rows) into a Markdown table."""
    if not table or not table[0]:
        return ""
    header, *rows = table
    header = [(c or "").strip() for c in header]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        cells = [(c or "").strip().replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def extract_pdf_sections(pdf_path):
    """
    Walk a PDF page by page, processing text lines AND tables in their true
    top-to-bottom reading order (not text-then-tables), so a table is tagged
    with the heading that actually precedes it on the page, not whichever
    heading happened to be last in the page's plain-text stream.
    Returns a list of {doc, section, page, text} dicts.
    """
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    sections = []
    current_heading = "Introduction"

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            found_tables = page.find_tables()
            table_bboxes = [t.bbox for t in found_tables]  # (x0, top, x1, bottom)

            def inside_a_table(top, bottom):
                return any(tb[1] <= top and bottom <= tb[3] for tb in table_bboxes)

            # Text lines, excluding any line that falls inside a table's bbox
            # (table cell text is handled separately via find_tables/extract).
            text_lines = page.extract_text_lines() or []
            events = []
            for line in text_lines:
                if inside_a_table(line["top"], line["bottom"]):
                    continue
                events.append({"top": line["top"], "kind": "text", "value": line["text"]})

            for t, bbox in zip(found_tables, table_bboxes):
                md_table = table_to_markdown(t.extract())
                if md_table:
                    events.append({"top": bbox[1], "kind": "table", "value": md_table})

            events.sort(key=lambda e: e["top"])

            for ev in events:
                if ev["kind"] == "text":
                    stripped = ev["value"].strip()
                    if HEADING_PATTERN.match(stripped):
                        current_heading = stripped
                        sections.append({"doc": doc_name, "section": current_heading, "page": page_num, "text": ""})
                        continue
                    if not sections or sections[-1]["section"] != current_heading:
                        sections.append({"doc": doc_name, "section": current_heading, "page": page_num, "text": ""})
                    sections[-1]["text"] += stripped + "\n"
                else:  # table
                    sections.append({
                        "doc": doc_name,
                        "section": f"{current_heading} (table)",
                        "page": page_num,
                        "text": ev["value"],
                        "_is_table": True,
                    })

    return sections


def sections_to_documents(sections):
    """Turn extracted sections into LangChain Documents, chunked where needed."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    documents = []

    for sec in sections:
        text = sec["text"].strip()
        if len(text) < 20:
            continue

        # Never split a Markdown table across chunks.
        if text.startswith("|") and "---" in text:
            chunks = [text]
        else:
            chunks = splitter.split_text(text)

        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={"doc": sec["doc"], "section": sec["section"], "page": sec["page"]},
            ))

    return documents


def main():
    pdf_paths = load_pdf_paths()

    all_documents = []
    for path in pdf_paths:
        print(f"Processing {os.path.basename(path)} ...")
        sections = extract_pdf_sections(path)
        docs = sections_to_documents(sections)
        print(f"  -> {len(docs)} chunks")
        all_documents.extend(docs)

    print(f"\nTotal chunks across all PDFs: {len(all_documents)}")
    print(f"Loading embedding model '{EMBEDDING_MODEL}' (runs locally, no API key needed) ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Building FAISS index ...")
    vectorstore = FAISS.from_documents(all_documents, embeddings)
    vectorstore.save_local(INDEX_DIR)

    print(f"\nDone. FAISS index saved to ./{INDEX_DIR}/ (index.faiss + index.pkl).")
    print("Commit this folder alongside app.py so the deployed app doesn't need to re-run ingestion.")


if __name__ == "__main__":
    main()
