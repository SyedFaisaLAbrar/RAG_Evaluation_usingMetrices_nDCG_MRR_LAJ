from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from typing import List
from dotenv import load_dotenv
import os

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
RETRIEVAL_K = 10

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model_name=MODEL)


from rag_core import get_latest_vector_store_path

def get_latest_vector_store_path(base_path: str = "vector_db") -> str:
    """Get the path to the most recent vector store."""
    import glob
    import os
    
    # First check for timestamped directories
    vector_dbs = glob.glob("vector_db_*")
    if vector_dbs:
        # Sort by modification time (newest first)
        vector_dbs.sort(key=os.path.getmtime, reverse=True)
        return vector_dbs[0]
    
    # Then check for base directory
    if os.path.exists(base_path):
        return base_path
    
    return None

def get_embedding_model_for_db(db_path: str):
    """Get the correct embedding model based on stored metadata."""
    # First try to read metadata.json
    metadata_file = os.path.join(db_path, "metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                model_name = metadata.get("embedding_model")
                
                if model_name:
                    print(f"Using embedding model from metadata: {model_name}")
                    if model_name.startswith("text-embedding"):
                        return OpenAIEmbeddings(model=model_name)
                    else:
                        return HuggingFaceEmbeddings(model_name=model_name)
        except:
            pass
    
    # Fallback to text file
    model_file = os.path.join(db_path, "embedding_model.txt")
    if os.path.exists(model_file):
        try:
            with open(model_file, 'r') as f:
                model_name = f.read().strip()
                print(f"Using embedding model from text file: {model_name}")
                if model_name.startswith("text-embedding"):
                    return OpenAIEmbeddings(model=model_name)
                else:
                    return HuggingFaceEmbeddings(model_name=model_name)
        except:
            pass
    
    # Last resort: try to detect
    print("WARNING: No metadata found, attempting to detect embedding model...")
    
    # Try HuggingFace first (most common)
    try:
        hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Just create the vectorstore to test, don't query
        test_vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=hf_embeddings
        )
        # If we can access the collection without error, it might work
        test_vectorstore._collection.count()
        print("Using HuggingFace embeddings (all-MiniLM-L6-v2)")
        return hf_embeddings
    except Exception as e:
        if "dimension" in str(e).lower() or "expected" in str(e).lower():
            print("HuggingFace embeddings failed, trying OpenAI...")
            try:
                openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                test_vectorstore = Chroma(
                    persist_directory=db_path,
                    embedding_function=openai_embeddings
                )
                test_vectorstore._collection.count()
                print("Using OpenAI embeddings")
                return openai_embeddings
            except Exception as e2:
                print(f"OpenAI also failed: {e2}")
    
    # If all else fails, raise an error
    raise ValueError(f"Could not determine embedding model for {db_path}")
    
def fetch_context(question: str, k: int = 10) -> List[Document]:
    """Fetch relevant context from vector store."""
    # Get the latest vector store path
    db_path = get_latest_vector_store_path()
    
    if not db_path or not os.path.exists(db_path):
        print(f"ERROR: No vector store found")
        return []
    
    try:
        # Get the appropriate embedding model for this vector store
        embeddings = get_embedding_model_for_db(db_path)
        
        # Load vector store with the CORRECT embeddings
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        
        # Perform similarity search
        docs = vectorstore.similarity_search(question, k=k)
        print(f"Retrieved {len(docs)} documents using {type(embeddings).__name__}")
        return docs
    except Exception as e:
        print(f"Error fetching context: {e}")
        return []


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
