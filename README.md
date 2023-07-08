# PdfGptIndexer

## Description
PdfGptIndexer is an efficient tool for indexing and searching PDF text data using OpenAI's GPT-2 model and FAISS (Facebook AI Similarity Search). This software is designed for rapid information retrieval and superior search accuracy.

## Libraries Used

1. [Textract](https://github.com/deanmalmgren/textract) - A Python library for extracting text from any document.
2. [Transformers](https://github.com/huggingface/transformers) - A library by Hugging Face providing state-of-the-art general-purpose architectures for Natural Language Understanding (NLU) and Natural Language Generation (NLG).
3. [Langchain](https://python.langchain.com/) - A text processing and embeddings library. 
4. [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss) - A library for efficient similarity search and clustering of dense vectors.

## Installing Dependencies

You can install all dependencies by running the following command:

```shell
pip install langchain openai textract transformers langchain faiss-cpu
```

## How It Works

The PdfGptIndexer operates in several stages:

1. It first processes a specified folder of PDF documents, extracting the text and splitting it into manageable chunks using a GPT-2 tokenizer from the Transformers library.
2. Each text chunk is then embedded using the OpenAI GPT-2 model through the LangChain library.
3. These embeddings are stored in a FAISS index, providing a compact and efficient storage method.
4. Finally, a query interface allows you to retrieve relevant information from the indexed data by asking questions. The application fetches and displays the most relevant text chunk.

![Untitled-2023-06-16-1537](https://github.com/raghavan/PdfGptIndexer/assets/131585/2e71dd82-bf4f-44db-b1ae-908cbb465deb)

## Advantages of Storing Embeddings Locally

Storing embeddings locally provides several advantages:

1. Speed: Once the embeddings are stored, retrieval of data is significantly faster as there's no need to compute embeddings in real-time.
2. Offline access: After the initial embedding creation, the data can be accessed offline.
3. Compute Savings: You only need to compute the embeddings once and reuse them, saving computational resources.
4. Scalability: This makes it feasible to work with large datasets that would be otherwise difficult to process in real-time.

## Running the Program

To run the program, you should:

1. Make sure you have installed all dependencies.
2. Clone the repository to your local machine.
3. Navigate to the directory containing the Python script.
4. Replace "<OPENAI_API_KEY>" with your actual OpenAI API key in the script.
5. Finally, run the script with Python.
```python
python3 pdf_gpt_indexer.py
```

Please ensure that the folders specified in the script for PDF documents and the output text files exist and are accessible. The query interface will start after the embeddings are computed and stored. You can exit the query interface by typing 'exit'.

## Exploring Custom Data with ChatGPT

Check out the post [here](https://devden.raghavan.studio/p/chatgpt-using-your-own-data) for a comprehensive guide on how to utilize ChatGPT with your own custom data.

