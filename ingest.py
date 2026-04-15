import os
import uuid
import warnings
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

warnings.filterwarnings('ignore')

load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / 'data'
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / 'chroma_medical_db')
COLLECTION_NAME = 'medical_documents'

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def main():
    print(f"Starting data ingestion from {DATA_DIR}...")
    
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        print("Data directory is empty or does not exist!")
        return

    loader = PyPDFDirectoryLoader(str(DATA_DIR))
    raw_pages = loader.load()
    
    for page_doc in raw_pages:
        stem = Path(page_doc.metadata.get('source', '')).stem
        page_doc.metadata.setdefault('category', 'medical_reference')
        page_doc.metadata['title'] = f"{stem} — p{page_doc.metadata.get('page', '?')}"
        page_doc.metadata['doc_id'] = str(uuid.uuid4())
        
    print(f"Loaded {len(raw_pages)} pages. Chunking now...")
    
    if not raw_pages:
        print("No pages loaded. Exiting.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=['\n\n', '\n', '. ', ' ', ''],
        length_function=len
    )
    
    chunks = splitter.split_documents(raw_pages)
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['chunk_length'] = len(chunk.page_content)
        
    print(f"Split into {len(chunks)} chunks. Building Chroma DB...")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    if Path(CHROMA_PERSIST_DIR).exists():
        import shutil
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print("Cleared existing ChromaDB to rebuild index.")
        
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )
    
    print(f"✅ Ingestion complete! DB built with {vectorstore._collection.count()} chunks.")

if __name__ == "__main__":
    main()
