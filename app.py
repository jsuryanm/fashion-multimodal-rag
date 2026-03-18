"""
app.py
=======
Gradio application — the entry point for the Fashion Style Finder UI.

Run with:
    python app.py

Then open http://localhost:5000 in your browser.

Startup sequence:
  1. Validate settings (Pydantic raises immediately on bad config)
  2. Check the FAISS index exists (tells user to run ingest.py if not)
  3. Load CLIP model + FAISS index into memory (once, at startup)
  4. Launch Gradio
"""

import logging
import os
import tempfile

import gradio as gr
from PIL import Image

from src.config.settings import settings
from src.models.embeddings import CLIPImageEmbedder
from src.models.vector_store import FashionVectorStore
from src.pipeline.rag_chain import FashionRAGChain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Application startup ────────────────────────────────────────────────────────

def build_chain() -> FashionRAGChain:
    """
    Load all heavy components once at startup.
    Raises RuntimeError with a clear message if prerequisites are missing.
    """
    if not FashionVectorStore.exists(settings.faiss_index_path):
        raise RuntimeError(
            f"FAISS index not found at '{settings.faiss_index_path}'.\n"
            "Run this first:  python ingest.py"
        )

    logger.info("Loading CLIP model...")
    embedder = CLIPImageEmbedder(settings.embeddings_model_name)

    logger.info("Loading FAISS index...")
    vector_store = FashionVectorStore.load(settings.faiss_index_path)

    logger.info("Building RAG chain...")
    chain = FashionRAGChain(embedder=embedder, vector_store=vector_store)

    logger.info("All components loaded. Ready.")
    return chain


# ── Gradio interface ───────────────────────────────────────────────────────────

def create_ui(chain: FashionRAGChain) -> gr.Blocks:
    """Build and return the Gradio Blocks interface."""

    def analyse(image: Image.Image) -> str:
        """
        Gradio event handler — receives a PIL Image from the Upload component
        and returns a Markdown analysis string.

        We pass the PIL Image directly to the chain. The chain's first step
        (_load_and_embed) accepts PIL Images natively.
        """
        if image is None:
            return "Please upload a fashion image to analyse."

        try:
            logger.info("Running analysis on uploaded image (size: %s)", image.size)
            result = chain.invoke(image)
            return result
        except Exception as exc:
            logger.error("Analysis error: %s", exc, exc_info=True)
            return f"An error occurred during analysis:\n\n`{exc}`\n\nPlease try again."

    # ── Layout ─────────────────────────────────────────────────────────────────
    with gr.Blocks(theme=gr.themes.Soft(), title="Fashion Style Finder") as demo:

        gr.Markdown(
            """
            # 👗 Fashion Style Finder
            Upload a fashion photo to get a detailed style analysis and matching catalogue items.

            **How it works:** Your image is embedded with CLIP, matched against the catalogue
            via FAISS similarity search, and analysed by a Groq vision LLM with the matched
            items as context.
            """
        )

        with gr.Row():

            # Left column: image upload + controls
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Fashion Image",
                    height=400,
                )
                analyse_btn = gr.Button("Analyse Style ✨", variant="primary", size="lg")
                status_md = gr.Markdown("_Ready. Upload an image and click Analyse._")

            # Right column: analysis output
            with gr.Column(scale=2):
                output_md = gr.Markdown(
                    label="Style Analysis",
                    value="_Analysis results will appear here._",
                    height=600,
                )

        # ── Example images (uses any images found in data/images/) ─────────────
        example_dir = settings.images_dir
        example_files = sorted(
            [str(p) for p in __import__("pathlib").Path(example_dir).glob("*.jpg")]
            + [str(p) for p in __import__("pathlib").Path(example_dir).glob("*.png")]
        )
        if example_files:
            gr.Markdown("### Example images from your catalogue")
            gr.Examples(
                examples=[[f] for f in example_files[:5]],
                inputs=[image_input],
                label="Click an example to load it",
            )

        # ── How it works section ───────────────────────────────────────────────
        with gr.Accordion(" How this works", open=False):
            gr.Markdown(
                """
                ### Pipeline Overview

                1. **CLIP Embedding** — Your image is converted to a 512-dimensional vector
                   using OpenAI's CLIP ViT-B/32 model via SentenceTransformers.

                2. **FAISS Search** — The vector is compared against all catalogue item
                   embeddings using cosine similarity (inner product on normalised vectors).
                   The top-5 most visually similar items are retrieved.

                3. **Prompt Construction** — A LangChain multimodal message is built
                   containing your image (as a base64 data URI) + the retrieved item list.

                4. **Groq LLM** — `llama-3.2-11b-vision-preview` on Groq analyses the image
                   and generates a structured Markdown fashion report.

                5. **Response Formatting** — Output is cleaned (escaped $, normalised headers)
                   for Gradio's Markdown renderer.

                ### Tech Stack
                `LangChain LCEL` · `Groq` · `CLIP (HuggingFace)` · `FAISS` · `Gradio`
                """
            )

        # ── Event handlers ─────────────────────────────────────────────────────
        analyse_btn.click(
            fn=lambda: "_Analysing image… this may take a few seconds._",
            inputs=None,
            outputs=status_md,
        ).then(
            fn=analyse,
            inputs=[image_input],
            outputs=output_md,
        ).then(
            fn=lambda: "_Analysis complete!_",
            inputs=None,
            outputs=status_md,
        )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        rag_chain = build_chain()
    except RuntimeError as e:
        print(f"\n Startup error:\n{e}\n")
        raise SystemExit(1)

    demo = create_ui(rag_chain)
    demo.launch(share=True)