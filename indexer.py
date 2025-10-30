import os
import sys
from pathlib import Path
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def index_pdfs(pdf_folder, index_path="faiss_index"):
    """
    Index all PDFs in a folder and save FAISS index to disk
    
    Args:
        pdf_folder: Path to folder containing PDF files
        index_path: Path where FAISS index will be saved
    """
    # Check if folder exists
    if not os.path.exists(pdf_folder):
        print(f"Error: Folder '{pdf_folder}' does not exist!")
        return False
    
    # Get all PDF files
    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{pdf_folder}'")
        return False
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Larger chunks for better context
        chunk_overlap=200,  # More overlap for continuity
        length_function=len,
    )
    
    # Process all PDFs
    all_documents = []
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}...")
        
        # Extract text
        text = extract_text_from_pdf(str(pdf_file))
        
        if not text.strip():
            print(f"  Warning: No text extracted from {pdf_file.name}")
            continue
        
        # Split into chunks with metadata
        chunks = text_splitter.create_documents(
            [text],
            metadatas=[{"source": pdf_file.name}]
        )
        
        all_documents.extend(chunks)
        print(f"  Created {len(chunks)} chunks")
    
    if not all_documents:
        print("No documents to index!")
        return False
    
    print(f"\nTotal chunks to index: {len(all_documents)}")
    print("Creating embeddings and building FAISS index...")
    
    # Create embeddings and FAISS index
    embeddings = OpenAIEmbeddings()
    
    try:
        vectorstore = FAISS.from_documents(all_documents, embeddings)
        
        # Save to disk
        vectorstore.save_local(index_path)
        print(f"\n✓ Successfully created and saved index to '{index_path}'")
        return True
        
    except Exception as e:
        print(f"\nError creating index: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("PDF Indexer - RAG System")
    print("=" * 60)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables!")
        print("Please create a .env file with: OPENAI_API_KEY=your_key_here")
        sys.exit(1)
    
    # Get PDF folder path from command line or use default
    if len(sys.argv) > 1:
        pdf_folder = sys.argv[1]
    else:
        pdf_folder = input("Enter path to PDF folder (default: ./pdf): ").strip()
        if not pdf_folder:
            pdf_folder = "./pdf"
    
    # Get index path from command line or use default
    if len(sys.argv) > 2:
        index_path = sys.argv[2]
    else:
        index_path = "faiss_index"
    
    # Index PDFs
    success = index_pdfs(pdf_folder, index_path)
    
    if success:
        print(f"\nYou can now use chatbot.py to query the indexed content!")
    else:
        print("\nIndexing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()