# PdfGptIndexer
**PdfGptIndexer was featured at the top of [Hacker News](https://news.ycombinator.com/item?id=36648794)!**
<img width="1139" alt="Screenshot 2024-05-18 at 9 38 18 AM" src="https://github.com/raghavan/raghavan/assets/131585/24215a9a-d423-45a8-8c4d-d9ee8b1ec752">

## Description
PdfGptIndexer is an efficient tool for indexing and querying PDF documents using OpenAI embeddings and FAISS (Facebook AI Similarity Search). It implements a RAG (Retrieval Augmented Generation) system that allows you to have intelligent conversations with your PDF documents. The software is designed for rapid information retrieval with superior search accuracy.

## How It Works

PdfGptIndexer consists of two main components:

### 1. **Indexer** (`indexer.py`) - One-time PDF Processing
The indexer processes your PDF documents and creates a searchable vector database:

1. **Extract Text**: Uses PyMuPDF to extract text from all PDF files in a folder
2. **Chunk Text**: Splits documents into manageable chunks (1000 characters with 200-character overlap) using LangChain's RecursiveCharacterTextSplitter
3. **Generate Embeddings**: Creates vector embeddings for each chunk using OpenAI's `text-embedding-ada-002` model
4. **Store Locally**: Saves the embeddings in a FAISS index on disk for fast retrieval

### 2. **Chatbot** (`chatbot.py`) - Interactive Q&A Interface
The chatbot provides an intelligent interface to query your indexed documents:

1. **Load Index**: Loads the pre-computed FAISS vector index from disk
2. **Semantic Search**: Converts your question into an embedding and finds the top 3 most similar document chunks
3. **Display Matches**: Shows you the similarity scores and text snippets from matched documents
4. **Generate Answer**: Uses GPT-4 to synthesize a coherent answer based on the retrieved context

![Untitled-2023-06-16-1537](https://github.com/raghavan/PdfGptIndexer/assets/131585/2e71dd82-bf4f-44db-b1ae-908cbb465deb)

## Advantages of Storing Embeddings Locally

Storing embeddings locally provides several key benefits:

1. **Speed**: Retrieval is significantly faster as embeddings are pre-computed—no need to regenerate them for each query
2. **Offline Access**: After initial creation, query your data without internet access to OpenAI (only the answer generation requires API calls)
3. **Cost Savings**: Compute embeddings once and reuse them, saving on API costs
4. **Scalability**: Makes it feasible to work with large document collections that would be expensive to process in real-time

## Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### 1. Installation

Clone the repository:
```bash
git clone https://github.com/raghavan/PdfGptIndexer.git
cd PdfGptIndexer
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install langchain langchain-openai langchain-community langchain-text-splitters openai pymupdf faiss-cpu python-dotenv tiktoken
```

### 2. Configuration

Create a `.env` file in the project root and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Prepare Your PDFs

Place your PDF files in the `pdf/` folder (or any folder of your choice).

## Usage

### Step 1: Index Your PDFs

Run the indexer to process your PDFs and create the vector database:

```bash
python3 indexer.py
```

Or specify a custom PDF folder:
```bash
python3 indexer.py /path/to/your/pdfs
```

Or specify both custom PDF folder and index location:
```bash
python3 indexer.py /path/to/your/pdfs /path/to/save/index
```

**What happens:**
- Extracts text from all PDFs in the folder
- Creates text chunks with metadata
- Generates embeddings using OpenAI
- Saves the FAISS index to `faiss_index/` (or your specified location)

**Note:** You only need to run this once, or when you add new PDFs to your collection.

### Step 2: Query Your Documents

Start the interactive chatbot:

```bash
python3 chatbot.py
```

Or specify a custom index location:
```bash
python3 chatbot.py /path/to/your/index
```
