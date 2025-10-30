import os
import sys
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Configuration
TOP_K = 3  # Number of similar documents to retrieve (configurable)
MODEL_NAME = "gpt-4"  # Default model

# Load environment variables
load_dotenv()

def load_vectorstore(index_path="faiss_index"):
    """Load FAISS index from disk"""
    if not os.path.exists(index_path):
        print(f"Error: Index not found at '{index_path}'")
        print("Please run indexer.py first to create the index!")
        return None
    
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            index_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception as e:
        print(f"Error loading index: {e}")
        return None

def get_similar_documents(vectorstore, query, k=TOP_K):
    """Find top K similar documents with scores"""
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    return docs_and_scores

def generate_answer(llm, query, context_docs):
    """Generate answer using LLM with retrieved context"""
    # Build context from documents
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question based on the provided context from PDF documents. If the context doesn't contain relevant information, say 'I don't know.'"),
        ("user", f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
    ])
    
    # Get response
    messages = prompt.format_messages()
    response = llm.invoke(messages)
    return response.content

def main():
    """Simple chatbot: query → retrieve → show scores → generate answer"""
    print("=" * 60)
    print("PDF RAG Chatbot")
    print("=" * 60)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found!")
        print("Create a .env file with: OPENAI_API_KEY=your_key")
        sys.exit(1)
    
    # Get index path
    index_path = sys.argv[1] if len(sys.argv) > 1 else "faiss_index"
    
    # Load vectorstore
    print(f"Loading index from '{index_path}'...")
    vectorstore = load_vectorstore(index_path)
    if vectorstore is None:
        sys.exit(1)
    
    # Initialize LLM
    print(f"Initializing {MODEL_NAME}...")
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.1)
    
    print(f"\n✓ Ready! Using {MODEL_NAME}, retrieving top {TOP_K} matches")
    print("Type 'exit' to quit")
    print("-" * 60)
    
    # Main loop
    while True:
        print("\n")
        query = input("You: ").strip()
        
        if not query:
            continue
        
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        try:
            # Step 1: Find similar documents
            docs_with_scores = get_similar_documents(vectorstore, query, k=TOP_K)
            
            # Step 2: Display matches with scores and text snippets
            print(f"\n🔍 Top {TOP_K} Similar Documents:")
            for i, (doc, score) in enumerate(docs_with_scores, 1):
                source = doc.metadata.get("source", "Unknown")
                status = "✅" if score < 0.4 else "⚠️" if score < 0.5 else "❌"
                
                # Get text snippet (first 200 chars)
                text_snippet = doc.page_content[:200].replace('\n', ' ')
                if len(doc.page_content) > 200:
                    text_snippet += "..."
                
                print(f"\n  {i}. {source} - Score: {score:.3f} {status}")
                print(f"     \"{text_snippet}\"")
            
            # Step 3: Generate answer using LLM
            context_docs = [doc for doc, score in docs_with_scores]
            print("\nBot: ", end="", flush=True)
            answer = generate_answer(llm, query, context_docs)
            print(answer)
        
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()