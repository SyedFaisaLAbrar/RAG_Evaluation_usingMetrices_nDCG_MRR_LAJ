import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
from dotenv import load_dotenv
import os
import tempfile
from pathlib import Path
import glob
import json
import time
from langchain_chroma import Chroma
from eval import evaluate_all_retrieval, evaluate_all_answers
from rag_core import (
    create_vector_store,
    load_documents,
    split_documents,
    get_embedding_model,
    visualize_embeddings,
    AVAILABLE_EMBEDDINGS,
    CHUNKING_STRATEGIES
)

load_dotenv(override=True)

# Color coding thresholds - Retrieval
MRR_GREEN = 0.9
MRR_AMBER = 0.75
NDCG_GREEN = 0.9
NDCG_AMBER = 0.75
COVERAGE_GREEN = 90.0
COVERAGE_AMBER = 75.0

# Color coding thresholds - Answer (1-5 scale)
ANSWER_GREEN = 4.5
ANSWER_AMBER = 4.0

def get_color(value: float, metric_type: str) -> str:
    """Get color based on metric value and type."""
    if metric_type == "mrr":
        if value >= MRR_GREEN:
            return "green"
        elif value >= MRR_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type == "ndcg":
        if value >= NDCG_GREEN:
            return "green"
        elif value >= NDCG_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type == "coverage":
        if value >= COVERAGE_GREEN:
            return "green"
        elif value >= COVERAGE_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type in ["accuracy", "completeness", "relevance"]:
        if value >= ANSWER_GREEN:
            return "green"
        elif value >= ANSWER_AMBER:
            return "orange"
        else:
            return "red"
    return "black"

def format_metric_html(
    label: str,
    value: float,
    metric_type: str,
    is_percentage: bool = False,
    score_format: bool = False,
) -> str:
    """Format a metric with color coding."""
    color = get_color(value, metric_type)
    if is_percentage:
        value_str = f"{value:.1f}%"
    elif score_format:
        value_str = f"{value:.2f}/5"
    else:
        value_str = f"{value:.4f}"
    return f"""
    <div style="margin: 10px 0; padding: 15px; background-color: #f8f9fa; border-radius: 8px; border-left: 5px solid {color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="font-size: 14px; color: #6c757d; margin-bottom: 5px; font-weight: 500;">{label}</div>
        <div style="font-size: 28px; font-weight: 600; color: {color};">{value_str}</div>
    </div>
    """

def create_embedding_config_section():
    """Create the embedding configuration section."""
    with gr.Group():
        gr.Markdown("### ⚙️ Embedding Model Configuration")
        gr.Markdown("Configure the embedding model and chunking strategy for the RAG system.")
        
        with gr.Row():
            with gr.Column(scale=1):
                enable_embedding = gr.Checkbox(
                    label="Enable Custom Embedding",
                    value=False,
                    info="Toggle to configure embedding settings"
                )
            
            with gr.Column(scale=3):
                gr.Markdown("""
                <div style="background-color: #e7f3ff; padding: 10px; border-radius: 5px; border-left: 4px solid #0066cc;">
                <strong>ℹ️ Note:</strong> Changes will rebuild the vector store. This may take a few minutes depending on document count.
                </div>
                """)
        
        with gr.Column(visible=False) as embedding_config:
            with gr.Row():
                with gr.Column(scale=1):
                    embedding_model = gr.Dropdown(
                        choices=list(AVAILABLE_EMBEDDINGS.keys()),
                        value=list(AVAILABLE_EMBEDDINGS.keys())[0],
                        label="Embedding Model",
                        info="Select the model for generating vector embeddings"
                    )
                    
                    model_info = gr.Markdown(
                        AVAILABLE_EMBEDDINGS[list(AVAILABLE_EMBEDDINGS.keys())[0]]["description"]
                    )
                
                with gr.Column(scale=1):
                    chunk_strategy = gr.Dropdown(
                        choices=list(CHUNKING_STRATEGIES.keys()),
                        value=list(CHUNKING_STRATEGIES.keys())[0],
                        label="Chunking Strategy",
                        info="Select document chunking approach"
                    )
                    
                    chunk_info = gr.Markdown(
                        CHUNKING_STRATEGIES[list(CHUNKING_STRATEGIES.keys())[0]]["description"]
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    chunk_size = gr.Slider(
                        minimum=256,
                        maximum=2048,
                        value=1000,
                        step=64,
                        label="Chunk Size",
                        info="Number of characters per chunk"
                    )
                
                with gr.Column(scale=1):
                    chunk_overlap = gr.Slider(
                        minimum=0,
                        maximum=400,
                        value=200,
                        step=20,
                        label="Chunk Overlap",
                        info="Overlap between consecutive chunks"
                    )
            
            with gr.Row():
                build_vectorstore_btn = gr.Button(
                    "Build Vector Store",
                    variant="primary",
                    size="lg"
                )
                
                build_status = gr.HTML("")
        
        # Update model info when selection changes
        def update_model_info(model):
            return AVAILABLE_EMBEDDINGS[model]["description"]
        
        embedding_model.change(
            fn=update_model_info,
            inputs=embedding_model,
            outputs=model_info
        )
        
        # Update chunk info when strategy changes
        def update_chunk_info(strategy):
            return CHUNKING_STRATEGIES[strategy]["description"]
        
        chunk_strategy.change(
            fn=update_chunk_info,
            inputs=chunk_strategy,
            outputs=chunk_info
        )
        
        # Toggle embedding config visibility
        enable_embedding.change(
            fn=lambda x: gr.update(visible=x),
            inputs=enable_embedding,
            outputs=embedding_config
        )
        
        return enable_embedding, embedding_model, chunk_strategy, chunk_size, chunk_overlap, build_vectorstore_btn, build_status

def create_visualization_section():
    """Create the vector store visualization section."""
    with gr.Group():
        gr.Markdown("### 📊Vector Store Visualization")
        gr.Markdown("Explore the semantic space of your document chunks in 2D and 3D.")
        
        with gr.Row():
            with gr.Column(scale=1):
                enable_viz = gr.Checkbox(
                    label="Enable Visualization",
                    value=False,
                    info="Toggle to visualize embeddings"
                )
            
            with gr.Column(scale=3):
                viz_method = gr.Radio(
                    choices=["t-SNE", "PCA"],
                    value="t-SNE",
                    label="Dimensionality Reduction Method",
                    info="Method to reduce high-dimensional vectors to 2D/3D"
                )
        
        with gr.Column(visible=False) as viz_section:
            with gr.Row():
                with gr.Column(scale=1):
                    viz_dim = gr.Radio(
                        choices=["2D", "3D"],
                        value="3D",
                        label="Visualization Dimension"
                    )
                    
                    perplexity = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=30,
                        step=5,
                        label="t-SNE Perplexity",
                        info="Balances local vs global aspects (t-SNE only)"
                    )
                    
                    refresh_viz_btn = gr.Button(
                        "🔄 Refresh Visualization",
                        variant="secondary"
                    )
                
                with gr.Column(scale=2):
                    viz_output = gr.Plot(label="Embedding Space Visualization")
            
            with gr.Row():
                stats_html = gr.HTML("")
        
        # Toggle visualization section
        enable_viz.change(
            fn=lambda x: gr.update(visible=x),
            inputs=enable_viz,
            outputs=viz_section
        )
        
        return enable_viz, viz_method, viz_dim, perplexity, refresh_viz_btn, viz_output, stats_html

def create_evaluation_section():
    """Create the evaluation section."""
    with gr.Group():
        gr.Markdown("### 📈 System Evaluation")
        gr.Markdown("Evaluate retrieval and answer generation quality.")
        
        with gr.Tabs():
            with gr.TabItem("🔍 Retrieval Evaluation"):
                with gr.Row():
                    retrieval_btn = gr.Button(
                        "Run Retrieval Evaluation",
                        variant="primary",
                        size="lg",
                        scale=1
                    )
                    
                    export_retrieval_btn = gr.Button(
                        "📥 Export Results",
                        variant="secondary",
                        scale=1
                    )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        retrieval_metrics = gr.HTML(
                            "<div style='padding: 20px; text-align: center; color: #6c757d; background: #f8f9fa; border-radius: 8px;'>Click 'Run Retrieval Evaluation' to start</div>"
                        )
                    
                    with gr.Column(scale=1):
                        retrieval_chart = gr.BarPlot(
                            x="Category",
                            y="Average MRR",
                            title="Average MRR by Document Category",
                            y_lim=[0, 1],
                            height=400,
                            color="Category",
                            # color_continuous_scale="Viridis"
                        )
            
            with gr.TabItem("💬 Answer Evaluation"):
                with gr.Row():
                    answer_btn = gr.Button(
                        "Run Answer Evaluation",
                        variant="primary",
                        size="lg",
                        scale=1
                    )
                    
                    export_answer_btn = gr.Button(
                        "📥 Export Results",
                        variant="secondary",
                        scale=1
                    )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        answer_metrics = gr.HTML(
                            "<div style='padding: 20px; text-align: center; color: #6c757d; background: #f8f9fa; border-radius: 8px;'>Click 'Run Answer Evaluation' to start</div>"
                        )
                    
                    with gr.Column(scale=1):
                        answer_chart = gr.BarPlot(
                            x="Category",
                            y="Average Accuracy",
                            title="Average Accuracy by Document Category",
                            y_lim=[1, 5],
                            height=400,
                            color="Category",
                            # color_continuous_scale="Viridis"
                        )
        
        return retrieval_btn, answer_btn, retrieval_metrics, answer_metrics, retrieval_chart, answer_chart, export_retrieval_btn, export_answer_btn

def run_retrieval_evaluation(progress=gr.Progress()):
    """Run retrieval evaluation and yield updates."""
    total_mrr = 0.0
    total_ndcg = 0.0
    total_coverage = 0.0
    category_mrr = defaultdict(list)
    count = 0

    for test, result, prog_value in evaluate_all_retrieval():
        count += 1
        total_mrr += result.mrr
        total_ndcg += result.ndcg
        total_coverage += result.keyword_coverage

        category_mrr[test.category].append(result.mrr)

        # Update progress bar only
        progress(prog_value, desc=f"Evaluating test {count}...")

    # Calculate final averages
    avg_mrr = total_mrr / count
    avg_ndcg = total_ndcg / count
    avg_coverage = total_coverage / count

    # Create final summary metrics HTML
    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Mean Reciprocal Rank (MRR)", avg_mrr, "mrr")}
        {format_metric_html("Normalized DCG (nDCG)", avg_ndcg, "ndcg")}
        {format_metric_html("Keyword Coverage", avg_coverage, "coverage", is_percentage=True)}
        <div style="margin-top: 20px; padding: 15px; background-color: #d4edda; border-radius: 8px; text-align: center; border: 1px solid #c3e6cb; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <span style="font-size: 16px; color: #155724; font-weight: 600;">✓ Evaluation Complete: {count} tests</span>
        </div>
    </div>
    """

    # Create final bar chart data
    category_data = []
    for category, mrr_scores in category_mrr.items():
        avg_cat_mrr = sum(mrr_scores) / len(mrr_scores)
        category_data.append({"Category": category.capitalize(), "Average MRR": avg_cat_mrr})

    df = pd.DataFrame(category_data)

    return final_html, df

def run_answer_evaluation(progress=gr.Progress()):
    """Run answer evaluation and yield updates."""
    total_accuracy = 0.0
    total_completeness = 0.0
    total_relevance = 0.0
    category_accuracy = defaultdict(list)
    count = 0

    for test, result, prog_value in evaluate_all_answers():
        count += 1
        total_accuracy += result.accuracy
        total_completeness += result.completeness
        total_relevance += result.relevance

        category_accuracy[test.category].append(result.accuracy)

        # Update progress bar only
        progress(prog_value, desc=f"Evaluating test {count}...")

    # Calculate final averages
    avg_accuracy = total_accuracy / count
    avg_completeness = total_completeness / count
    avg_relevance = total_relevance / count

    # Create final summary metrics HTML
    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Accuracy", avg_accuracy, "accuracy", score_format=True)}
        {format_metric_html("Completeness", avg_completeness, "completeness", score_format=True)}
        {format_metric_html("Relevance", avg_relevance, "relevance", score_format=True)}
        <div style="margin-top: 20px; padding: 15px; background-color: #d4edda; border-radius: 8px; text-align: center; border: 1px solid #c3e6cb; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <span style="font-size: 16px; color: #155724; font-weight: 600;">✓ Evaluation Complete: {count} tests</span>
        </div>
    </div>
    """

    # Create final bar chart data
    category_data = []
    for category, accuracy_scores in category_accuracy.items():
        avg_cat_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        category_data.append({"Category": category.capitalize(), "Average Accuracy": avg_cat_accuracy})

    df = pd.DataFrame(category_data)

    return final_html, df

def build_vector_store(embedding_model, chunk_strategy, chunk_size, chunk_overlap, progress=gr.Progress()):
    """Build the vector store with specified configuration."""
    try:
        progress(0.1, desc="Loading documents...")
        documents = load_documents()
        print(f"Loaded {len(documents)} documents")  # Debug print
        
        if len(documents) == 0:
            return """
            <div style="padding: 15px; background-color: #f8d7da; border-radius: 8px; border: 1px solid #f5c6cb;">
                <span style="color: #721c24; font-weight: 600;">❌ No documents found in knowledge-base/</span>
            </div>
            """
        
        progress(0.3, desc="Splitting documents...")
        chunks = split_documents(documents, chunk_strategy, chunk_size, chunk_overlap)
        print(f"Created {len(chunks)} chunks")  # Debug print
        
        if len(chunks) == 0:
            return """
            <div style="padding: 15px; background-color: #f8d7da; border-radius: 8px; border: 1px solid #f5c6cb;">
                <span style="color: #721c24; font-weight: 600;">❌ No chunks created from documents</span>
            </div>
            """
        
        progress(0.6, desc="Creating embeddings...")
        embeddings = get_embedding_model(embedding_model)
        
        progress(0.8, desc="Building vector store...")
        
        # Create a timestamped directory name
        import time
        timestamp = int(time.time())
        db_directory = f"vector_db_{timestamp}"
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_directory
        )

        import json
        metadata = {
            "embedding_model": embedding_model,
            "embedding_dimensions": get_embedding_dimensions(embedding_model),
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "timestamp": timestamp,
            "num_chunks": len(chunks)
        }
        
        with open(os.path.join(db_directory, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Also save a simple text file for easy reading
        with open(os.path.join(db_directory, "embedding_model.txt"), "w") as f:
            f.write(embedding_model)
        
        # Update active vector store pointer
        with open("active_vector_store.txt", "w") as f:
            f.write(db_directory)
            
        with open("latest_vector_db.txt", "w") as f:
            f.write(db_directory)
            
        # No need to call .persist() - it's automatic
        
        # Verify it was created
        if os.path.exists(db_directory):
            files = os.listdir(db_directory)
            print(f"Vector store directory created with files: {files}")  # Debug print
            
            # Count vectors in the collection
            collection = vectorstore._collection
            count = collection.count()
            print(f"Vector store contains {count} vectors")
            
            # Create or update symlink for consistency
            if os.path.exists("vector_db"):
                if os.path.islink("vector_db"):
                    os.unlink("vector_db")
                else:
                    import shutil
                    shutil.rmtree("vector_db")
            
            # On Windows, use junction instead of symlink if needed
            try:
                os.symlink(db_directory, "vector_db", target_is_directory=True)
            except (OSError, AttributeError):
                # Fallback: just note the directory name
                pass
        
        return f"""
        <div style="padding: 15px; background-color: #d4edda; border-radius: 8px; border: 1px solid #c3e6cb;">
            <span style="color: #155724; font-weight: 600;">✅ Vector store built successfully!</span><br>
            <span style="color: #155724;">Model: {embedding_model}</span><br>
            <span style="color: #155724;">Documents: {len(documents)}</span><br>
            <span style="color: #155724;">Chunks: {len(chunks)}</span><br>
            <span style="color: #155724;">Vectors: {vectorstore._collection.count()}</span><br>
            <span style="color: #155724;">Location: {db_directory}</span>
        </div>
        """
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print full error
        return f"""
        <div style="padding: 15px; background-color: #f8d7da; border-radius: 8px; border: 1px solid #f5c6cb;">
            <span style="color: #721c24; font-weight: 600;">❌ Error building vector store:</span><br>
            <span style="color: #721c24;">{str(e)}</span>
        </div>
        """
        
def refresh_visualization(viz_method, viz_dim, perplexity, progress=gr.Progress()):
    """Refresh the embedding visualization."""
    try:
        progress(0.2, desc="Loading vectors...")
        fig, stats = visualize_embeddings(
            method=viz_method.lower(),
            dim=2 if viz_dim == "2D" else 3,
            perplexity=perplexity if viz_method == "t-SNE" else None
        )
        
        progress(0.8, desc="Rendering...")
        
        if stats.get('count', 0) == 0:
            stats_html = f"""
            <div style="padding: 15px; background-color: #fff3cd; border-radius: 8px; border: 1px solid #ffeeba;">
                <span style="color: #856404; font-weight: 600;">⚠️ {stats.get('error', 'No vectors to visualize')}</span><br>
                <span style="color: #856404;">Please build a vector store first in the Configuration tab.</span>
            </div>
            """
        else:
            stats_html = f"""
            <div style="padding: 10px; background-color: #f8f9fa; border-radius: 8px;">
                <h4 style="margin-top: 0;">Vector Store Statistics</h4>
                <ul style="list-style-type: none; padding: 0;">
                    <li><strong>Total Vectors:</strong> {stats['count']:,}</li>
                    <li><strong>Dimensions:</strong> {stats['dimensions']:,}</li>
                    <li><strong>Document Types:</strong> {', '.join(stats['doc_types']) if stats['doc_types'] else 'None'}</li>
                </ul>
            </div>
            """
        
        return fig, stats_html
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig, f"""
        <div style="padding: 15px; background-color: #f8d7da; border-radius: 8px;">
            <span style="color: #721c24;">❌ Visualization error: {str(e)}</span><br>
            <span style="color: #721c24;">Please ensure you have built a vector store first.</span>
        </div>
        """

def get_embedding_dimensions(model_name: str) -> int:
    """Get the dimensions for different embedding models."""
    dimensions = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "BAAI/bge-small-en-v1.5": 384
    }
    return dimensions.get(model_name, 768)
    
def export_results(df, metric_type):
    """Export evaluation results to CSV."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        return f.name

def main():
    """Launch the Gradio evaluation app."""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
        font=["Inter", "system-ui", "sans-serif"]
    )

    with gr.Blocks(title="Insurellm RAG Evaluation Dashboard") as app:
        
        gr.Markdown("""
        # 🔬 Insurellm RAG System Evaluation Dashboard
        
        A comprehensive evaluation platform for Retrieval-Augmented Generation systems. 
        This dashboard enables researchers and practitioners to:
        - **Configure** embedding models and chunking strategies
        - **Visualize** the semantic space of document embeddings
        - **Evaluate** retrieval and answer generation quality
        
        *Built for the research community • Version 1.0*
        """)
        
        with gr.Tabs() as tabs:
            with gr.TabItem("⚙️ Configuration", id="config_tab"):
                embedding_toggle, embedding_model, chunk_strategy, chunk_size, chunk_overlap, build_btn, build_status = create_embedding_config_section()
            
            with gr.TabItem("📊 Visualization", id="viz_tab"):
                viz_toggle, viz_method, viz_dim, perplexity, refresh_viz_btn, viz_output, viz_stats = create_visualization_section()
            
            with gr.TabItem("📈 Evaluation", id="eval_tab"):
                retrieval_btn, answer_btn, retrieval_metrics, answer_metrics, retrieval_chart, answer_chart, export_retrieval_btn, export_answer_btn = create_evaluation_section()
        
        # Wire up the build vector store button
        build_btn.click(
            fn=build_vector_store,
            inputs=[embedding_model, chunk_strategy, chunk_size, chunk_overlap],
            outputs=build_status
        )
        
        # Wire up the refresh visualization button
        refresh_viz_btn.click(
            fn=refresh_visualization,
            inputs=[viz_method, viz_dim, perplexity],
            outputs=[viz_output, viz_stats]
        )
        
        # Wire up the retrieval evaluation
        retrieval_btn.click(
            fn=run_retrieval_evaluation,
            outputs=[retrieval_metrics, retrieval_chart]
        )
        
        # Wire up the answer evaluation
        answer_btn.click(
            fn=run_answer_evaluation,
            outputs=[answer_metrics, answer_chart]
        )
        
        # Wire up export buttons
        # export_retrieval_btn.click(
        #     fn=lambda: export_results(retrieval_chart.value, "retrieval"),
        #     inputs=[],
        #     outputs=gr.File(label="Download CSV")
        # )
        
        # export_answer_btn.click(
        #     fn=lambda: export_results(answer_chart.value, "answer"),
        #     inputs=[],
        #     outputs=gr.File(label="Download CSV")
        # )
        
        # Add footer
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #6c757d; padding: 20px;">
            <p>🏢 Insurellm Research • Released under MIT License • 
            <a href="https://github.com/syedfaisalabrar" target="_blank">GitHub Repository</a></p>
            <p style="font-size: 12px;">For questions or collaborations, contact syedfaisalabrar100@gmail.com</p>
        </div>
        """)

    app.launch(inbrowser=True, theme=theme, css="""
        .gradio-container {
            max-width: 1400px !important;
            margin: auto !important;
        }
        .prose {
            color: #2c3e50;
        }
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700 !important;
        }
        .group {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            background: white;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        }
    """)


if __name__ == "__main__":
    main()