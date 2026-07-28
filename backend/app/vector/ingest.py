import os
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.app.config import settings
from backend.app.core.logger import logger
from backend.app.vector.store import VectorStoreManager

def ingest_policies_pdfs():
    """Finds all policy PDFs in Dataset folder, chunks them, and uploads them to ChromaDB."""
    dataset_dir = settings.DATASET_DIR
    pdf_files = list(dataset_dir.glob("*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {dataset_dir}")
        
    logger.info(f"Found {len(pdf_files)} PDF files to process.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    all_texts = []
    all_metadatas = []
    all_ids = []
    
    for pdf_path in pdf_files:
        doc_name = pdf_path.stem
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        try:
            reader = PdfReader(pdf_path)
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                text = page.extract_text()
                if not text or not text.strip():
                    logger.warning(f"No text found on page {page_num} of {pdf_path.name}")
                    continue
                
                # Split page text into chunks
                chunks = text_splitter.split_text(text)
                for chunk_idx, chunk in enumerate(chunks):
                    all_texts.append(chunk)
                    all_metadatas.append({
                        "source": pdf_path.name,
                        "document_name": doc_name,
                        "page": page_num,
                        "chunk_index": chunk_idx
                    })
                    all_ids.append(f"{doc_name}_p{page_num}_c{chunk_idx}")
                    
        except Exception as e:
            logger.error(f"Error reading {pdf_path.name}: {str(e)}")
            raise e
            
    if all_texts:
        logger.info(f"Total chunks extracted: {len(all_texts)}. Indexing in vector DB...")
        vstore = VectorStoreManager()
        vstore.reset_collection()
        vstore.add_documents(all_texts, all_metadatas, all_ids)
        logger.info("PDF policy files ingested successfully into vector database.")
    else:
        logger.warning("No text was extracted from any PDF documents.")

if __name__ == "__main__":
    ingest_policies_pdfs()
