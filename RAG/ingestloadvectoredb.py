
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# 1. Load environment variables from .env
# --------------------------------------------------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

PDF_DATA_DIR = BASE_DIR / os.getenv("PDF_DATA_DIR", "data")
FAISS_INDEX_PATH = BASE_DIR / os.getenv(
    "FAISS_INDEX_PATH", "faiss_index"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdf_documents():
    """Load text and metadata from all PDFs in the data folder."""

    if not PDF_DATA_DIR.exists():
        raise FileNotFoundError(
            f"PDF folder not found: {PDF_DATA_DIR}"
        )

    pdf_files = sorted(PDF_DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PDF_DATA_DIR}. "
            "Please add agriculture scheme PDFs."
        )

    all_documents = []

    for pdf_file in pdf_files:
        print(f"\nLoading PDF: {pdf_file.name}")

        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            # Ignore pages without meaningful text.
            documents = [
                doc for doc in documents
                if doc.page_content.strip()
            ]

            # Preserve the source filename and page number.
            for doc in documents:
                doc.metadata["source"] = pdf_file.name
                doc.metadata["page"] = doc.metadata.get("page", 0) + 1

            if documents:
                all_documents.extend(documents)
                print(f"  Pages loaded: {len(documents)}")
            else:
                print(
                    "  Warning: No extractable text found "
                    "in this PDF."
                )

        except Exception as error:
            print(f"  Error reading {pdf_file.name}: {error}")

    if not all_documents:
        raise ValueError(
            "No readable PDF text was found. "
            "Check whether the PDFs contain selectable text."
        )

    print(f"\nTotal pages loaded: {len(all_documents)}")
    return all_documents


def split_documents(documents):
    """Split PDF text into smaller, searchable chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    print(f"Total text chunks created: {len(chunks)}")
    return chunks


def create_faiss_database(chunks):
    """Generate local embeddings and save the FAISS index."""

    print("\nLoading the embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Generating embeddings and building FAISS index...")

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)

    vector_db.save_local(str(FAISS_INDEX_PATH))

    print("\nFAISS database created successfully!")
    print(f"Index location: {FAISS_INDEX_PATH.resolve()}")
    print(f"Indexed chunks: {len(chunks)}")


def main():
    print("=" * 55)
    print(" AGRICULTURE SCHEMES - PDF INGESTION")
    print("=" * 55)

    try:
        documents = load_pdf_documents()
        chunks = split_documents(documents)
        create_faiss_database(chunks)

        print("\nIngestion completed successfully!")

    except Exception as error:
        print(f"\nIngestion failed: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
