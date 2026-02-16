# ingest_docs.py
import os
from pathlib import Path
from typing import List, Tuple
import psycopg
from sentence_transformers import SentenceTransformer

# Optional readers
from pypdf import PdfReader
from docx import Document as DocxDocument

DB_CONFIG = {
    "dbname": "chatbot_rag",
    "user": "postgres",
    "password": "t17@ACHA",
    "host": "localhost",
    "port": "5432",
}

DATA_DIR = Path("data")  # mets tes fichiers ici: data/*.pdf, *.docx, *.txt
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def connect():
    conn_string = (
        f"dbname={DB_CONFIG['dbname']} user={DB_CONFIG['user']} "
        f"password={DB_CONFIG['password']} host={DB_CONFIG['host']} port={DB_CONFIG['port']}"
    )
    return psycopg.connect(conn_string)

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)

def read_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def load_text(path: Path) -> Tuple[str, str]:
    ext = path.suffix.lower()
    if ext == ".txt":
        return read_txt(path), "txt"
    if ext == ".pdf":
        return read_pdf(path), "pdf"
    if ext == ".docx":
        return read_docx(path), "docx"
    raise ValueError(f"Format non supporté: {ext}")

def normalize(text: str) -> str:
    # ✅ enlever les NUL bytes (cause du crash PostgreSQL)
    text = text.replace("\x00", " ")

    text = text.replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n"))
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if len(chunk) > 40:
            chunks.append(chunk)

        # ✅ si on est arrivé à la fin, on sort
        if end >= n:
            break

        # ✅ avance garantissant le progrès
        next_start = end - overlap
        if next_start <= start:
            next_start = end  # fallback: pas d'overlap si ça bloque
        start = next_start

    return chunks

def insert_document(cur, title: str, source_type: str, source_path: str) -> int:
    cur.execute(
        "INSERT INTO documents (title, source_type, source_path) VALUES (%s, %s, %s) RETURNING id",
        (title, source_type, source_path),
    )
    return cur.fetchone()[0]

def insert_chunk(cur, doc_id: int, chunk_index: int, content: str, embedding: List[float]):
    cur.execute(
        "INSERT INTO chunks (doc_id, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
        (doc_id, chunk_index, content, embedding),
    )

def main():
    if not DATA_DIR.exists():
        print(f"❌ Dossier introuvable: {DATA_DIR.resolve()}")
        print("➡️ Crée un dossier 'data' et mets tes documents dedans.")
        return

    model = SentenceTransformer(MODEL_NAME)

    files = [p for p in DATA_DIR.iterdir() if p.suffix.lower() in [".pdf", ".docx", ".txt"]]
    if not files:
        print("❌ Aucun document trouvé dans /data (pdf/docx/txt).")
        return

    with connect() as conn:
        with conn.cursor() as cur:
            for path in files:
                print(f"\n📄 Import: {path.name}")
                raw, stype = load_text(path)
                text = normalize(raw)

                chunks = chunk_text(text)
                if not chunks:
                    print("⚠️ Texte vide, ignoré.")
                    continue

                doc_id = insert_document(cur, title=path.stem, source_type=stype, source_path=str(path))
                print(f"✅ Document id={doc_id} | chunks={len(chunks)}")

                # embeddings batch (plus rapide)
                embs = model.encode(chunks, show_progress_bar=False)
                for idx, (chunk, emb) in enumerate(zip(chunks, embs), start=1):
                    insert_chunk(cur, doc_id, idx, chunk, emb.tolist())

                conn.commit()

    print("\n🎉 Import terminé.")

if __name__ == "__main__":
    main()