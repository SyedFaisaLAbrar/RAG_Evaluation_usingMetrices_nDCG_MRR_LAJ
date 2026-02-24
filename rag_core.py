import os
import glob
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings

# Available embedding models with descriptions
AVAILABLE_EMBEDDINGS = {
    "all-MiniLM-L6-v2": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Lightweight model (80MB) optimized for speed. 384-dimensional vectors. Good balance of quality and performance."
    },
    "all-mpnet-base-v2": {
        "model": "sentence-transformers/all-mpnet-base-v2",
        "description": "High-quality model (420MB) with 768-dimensional vectors. Best for semantic search accuracy."
    },
    "text-embedding-3-small": {
        "model": "text-embedding-3-small",
        "description": "OpenAI's efficient embedding model. 1536-dimensional vectors. Requires API key."
    },
    "text-embedding-3-large": {
        "model": "text-embedding-3-large",
        "description": "OpenAI's most capable embedding model. 3072-dimensional vectors. Highest quality but API-based."
    },
    "BAAI/bge-small-en-v1.5": {
        "model": "BAAI/bge-small-en-v1.5",
        "description": "State-of-the-art small model (33MB). 384-dimensional vectors. Excellent for retrieval tasks."
    }
}

# Available chunking strategies
CHUNKING_STRATEGIES = {
    "recursive_character": {
        "name": "Recursive Character Splitter",
        "description": "Splits text recursively by characters. Preserves natural boundaries like paragraphs and sentences."
    },
    "token_count": {
        "name": "Token-based Splitter",
        "description": "Splits based on token count using tiktoken. Better for controlling API token usage."
    },
    "semantic": {
        "name": "Semantic Chunking",
        "description": "Attempts to preserve semantic boundaries using sentence embeddings. More computationally intensive."
    }
}

def get_embedding_model(model_name: str):
    """Get the embedding model instance based on selection."""
    model_config = AVAILABLE_EMBEDDINGS[model_name]
    
    if model_name.startswith("text-embedding"):
        return OpenAIEmbeddings(model=model_config["model"])
    else:
        return HuggingFaceEmbeddings(
            model_name=model_config["model"],
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

def load_documents() -> List[Any]:
    """Load all documents from the knowledge base."""
    documents = []
    folders = glob.glob("knowledge-base/*")
    
    for folder in folders:
        if os.path.isdir(folder):
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(
                folder,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            folder_docs = loader.load()
            for doc in folder_docs:
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)
    
    return documents

def get_latest_vector_store_path(base_path: str = "vector_db") -> str:
    """
    Get the path to the most recent vector store.
    Returns the path if found, otherwise returns the base path.
    """
    import glob
    import os
    
    # First check for timestamped directories
    vector_dbs = glob.glob("vector_db_*")
    if vector_dbs:
        # Sort by modification time (newest first)
        vector_dbs.sort(key=os.path.getmtime, reverse=True)
        latest = vector_dbs[0]
        print(f"Found timestamped vector store: {latest}")
        return latest
    
    # Then check for base directory
    if os.path.exists(base_path):
        print(f"Using base vector store: {base_path}")
        return base_path
    
    print(f"No vector store found at {base_path} or vector_db_*")
    return base_path
    
def split_documents(documents: List[Any], strategy: str, chunk_size: int, chunk_overlap: int) -> List[Any]:
    """Split documents using the specified strategy."""
    if strategy == "recursive_character":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    elif strategy == "token_count":
        from langchain_text_splitters import TokenTextSplitter
        text_splitter = TokenTextSplitter(
            chunk_size=chunk_size // 4,  # Approximate token to character ratio
            chunk_overlap=chunk_overlap // 4
        )
    else:  # semantic - fallback to recursive for now
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    return text_splitter.split_documents(documents)

def create_vector_store(chunks: List[Any], embeddings, persist_dir: str) -> Chroma:
    """Create or update a vector store."""
    # Clear existing if any
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)
    
    print(f"Creating vector store in {persist_dir} with {len(chunks)} chunks")  # Debug
    
    # Create new vector store - persist is automatic
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    # No need for explicit persist - it's automatic in newer versions
    
    # Verify it was created
    if os.path.exists(persist_dir):
        print(f"Vector store created successfully. Contents: {os.listdir(persist_dir)}")
        # Check if there are vectors
        try:
            count = vectorstore._collection.count()
            print(f"Vector store contains {count} vectors")
        except:
            print("Could not count vectors")
    else:
        print(f"ERROR: Vector store directory {persist_dir} was not created!")
    
    return vectorstore

def visualize_embeddings(
    method: str = "tsne",
    dim: int = 3,
    perplexity: Optional[int] = 30,
    db_path: str = "vector_db"
) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    Visualize embeddings using dimensionality reduction.
    """
    actual_db_path = get_latest_vector_store_path(db_path)
    
    if not actual_db_path or not os.path.exists(actual_db_path):
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No vector store found. Please build one first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig, {'error': 'No vector store found'}
    
    try:
        # Get the CORRECT embedding model for this vector store
        embeddings = get_embedding_model_for_db(actual_db_path)
        
        # Load vector store with correct embeddings
        vectorstore = Chroma(
            persist_directory=actual_db_path,
            embedding_function=embeddings
        )
        
        # Get all vectors and metadata
        collection = vectorstore._collection
        result = collection.get(include=['embeddings', 'documents', 'metadatas'])
        
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        metadatas = result['metadatas']
        
        # Check if vectors exist
        if len(vectors) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No vectors found in the database. The vector store might be empty.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            stats = {
                'count': 0,
                'dimensions': 0,
                'doc_types': [],
                'method': method,
                'reduced_dim': dim,
                'error': 'Empty vector store'
            }
            return fig, stats
        
        # Check if vectors have the right shape
        if len(vectors.shape) == 1:
            # Reshape if it's a 1D array
            vectors = vectors.reshape(-1, 1)
        
        # Extract document types for coloring
        doc_types = [metadata['doc_type'] for metadata in metadatas]
        unique_types = list(set(doc_types))
        color_map = px.colors.qualitative.Plotly
        colors = [color_map[unique_types.index(t) % len(color_map)] for t in doc_types]
        
        # Reduce dimensionality
        if method == "tsne":
            # Adjust perplexity based on sample size
            n_samples = len(vectors)
            adjusted_perplexity = min(perplexity or 30, n_samples - 1) if n_samples > 1 else 1
            
            if n_samples < 2:
                fig = go.Figure()
                fig.add_annotation(
                    text="Need at least 2 samples for t-SNE visualization.",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16)
                )
                stats = {
                    'count': n_samples,
                    'dimensions': vectors.shape[1] if len(vectors.shape) > 1 else 1,
                    'doc_types': unique_types,
                    'method': method,
                    'reduced_dim': dim,
                    'error': 'Insufficient samples'
                }
                return fig, stats
            
            reducer = TSNE(
                n_components=dim,
                random_state=42,
                perplexity=adjusted_perplexity
            )
        else:  # PCA
            if len(vectors) == 0:
                reducer = PCA(n_components=min(dim, 1), random_state=42)
            else:
                n_components = min(dim, len(vectors), vectors.shape[1] if len(vectors.shape) > 1 else 1)
                reducer = PCA(n_components=n_components, random_state=42)
        
        # Fit transform with error handling
        try:
            reduced_vectors = reducer.fit_transform(vectors)
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Dimensionality reduction failed: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            stats = {
                'count': len(vectors),
                'dimensions': vectors.shape[1] if len(vectors.shape) > 1 else 1,
                'doc_types': unique_types,
                'method': method,
                'reduced_dim': dim,
                'error': str(e)
            }
            return fig, stats
        
        # Create hover text
        hover_texts = [
            f"Type: {t}<br>Text: {d[:150]}..." if len(d) > 150 else f"Type: {t}<br>Text: {d}"
            for t, d in zip(doc_types, documents)
        ]
        
        # Create plot
        if dim == 2:
            fig = go.Figure(data=[go.Scatter(
                x=reduced_vectors[:, 0],
                y=reduced_vectors[:, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    color=colors,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=hover_texts,
                hoverinfo='text'
            )])
            
            fig.update_layout(
                title='2D Embedding Space Visualization',
                xaxis_title=f'{method.upper()} Component 1',
                yaxis_title=f'{method.upper()} Component 2',
                width=800,
                height=600,
                plot_bgcolor='rgba(240,240,240,0.95)',
                paper_bgcolor='white',
                font=dict(family="Inter", size=12)
            )
        else:
            fig = go.Figure(data=[go.Scatter3d(
                x=reduced_vectors[:, 0],
                y=reduced_vectors[:, 1],
                z=reduced_vectors[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=colors,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=hover_texts,
                hoverinfo='text'
            )])
            
            fig.update_layout(
                title='3D Embedding Space Visualization',
                scene=dict(
                    xaxis_title=f'{method.upper()} Component 1',
                    yaxis_title=f'{method.upper()} Component 2',
                    zaxis_title=f'{method.upper()} Component 3',
                    bgcolor='rgba(240,240,240,0.95)'
                ),
                width=900,
                height=700,
                paper_bgcolor='white',
                font=dict(family="Inter", size=12)
            )
        
        # Add legend for document types
        for i, doc_type in enumerate(unique_types):
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color_map[i % len(color_map)]),
                name=doc_type.capitalize(),
                showlegend=True
            ))
        
        # Statistics
        stats = {
            'count': len(vectors),
            'dimensions': vectors.shape[1] if len(vectors.shape) > 1 else 1,
            'doc_types': unique_types,
            'method': method,
            'reduced_dim': dim
        }
        stats['path'] = actual_db_path
        
        return fig, stats
        
    except Exception as e:
        # Error handling
        fig = go.Figure()
        fig.add_annotation(
            text=f"Visualization error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        stats = {
            'count': 0,
            'dimensions': 0,
            'doc_types': [],
            'method': method,
            'reduced_dim': dim,
            'error': str(e)
        }
        return fig, stats