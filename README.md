# 👗 Fashion Style Finder

A multimodal RAG (Retrieval-Augmented Generation) app that analyses fashion images and returns a detailed style breakdown with matching catalogue items. Upload any clothing photo and get garment descriptions, style categorisation, and shoppable product links — powered by CLIP embeddings, FAISS vector search, and a Groq vision LLM.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.0%2B-green)
![Groq](https://img.shields.io/badge/LLM-Groq-orange)
![FAISS](https://img.shields.io/badge/VectorStore-FAISS-red)
![Gradio](https://img.shields.io/badge/UI-Gradio-yellow)

---

## How It Works

The pipeline runs in 5 steps every time you upload an image:

```
Upload image
     │
     ▼
┌─────────────────────────────────────────────┐
│ Step 1 — CLIP Embedding                     │
│ Image → 512-dim vector via clip-ViT-B-32    │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ Step 2 — FAISS Similarity Search            │
│ Query vector → top-5 most similar items     │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ Step 3 — Prompt Construction                │
│ Image (base64) + retrieved items → messages │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ Step 4 — Groq Vision LLM                    │
│ llama-3.2-11b-vision-preview generates      │
│ structured Markdown fashion analysis        │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ Step 5 — Response Formatting                │
│ Clean Markdown → Gradio display             │
└─────────────────────────────────────────────┘
```

**Why this approach?**
- **CLIP** understands fashion semantics ("red plaid blazer") rather than just pixel patterns, making similarity search meaningful
- **FAISS** runs the similarity search locally — fast, free, no API calls
- **RAG** lets the LLM see both the image AND structured catalogue context it would otherwise not have access to
- **Groq** delivers sub-3s LLM responses — essential for a real-time UI

---

## Project Structure

```
fashion-finder/
├── app.py                        # Gradio UI — entry point
├── ingest.py                     # One-time index builder — run before app.py
├── requirements.txt
├── .env.example                  # Config template — copy to .env
├── .gitignore
│
├── config/
│   └── settings.py               # All config via Pydantic Settings + .env
│
├── data/
│   ├── images/                   # ← Put your 5 images here
│   └── local_dataset_adapter.py  # ← Edit CATALOGUE here to describe your images
│
├── models/
│   ├── embeddings.py             # CLIP image embedder (SentenceTransformers)
│   ├── vector_store.py           # FAISS index — store, search, save, load
│   └── llm_service.py            # Groq ChatLLM + multimodal prompt builder
│
├── pipeline/
│   └── rag_chain.py              # 5-step LangChain LCEL RAG chain
│
└── utils/
    ├── image_utils.py            # load_image, pil_to_data_uri
    └── response_formatter.py     # Markdown cleanup for Gradio display
```

---

## Quickstart

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com) (no credit card required)
- 5 fashion images (downloaded from Google or AI-generated)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the CLIP model (~350MB) from HuggingFace Hub. It's cached locally after that.

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and paste your Groq API key:

```env
GROQ_API_KEY=gsk_your_key_here
```

### 3. Add your images

```bash
mkdir -p data/images
```

Copy your 5 images into `data/images/` named `image1.jpg` through `image5.jpg`. They should be visually distinct — for example:

| File | Suggested content |
|------|-------------------|
| `image1.jpg` | Floral dress outfit |
| `image2.jpg` | Business suit |
| `image3.jpg` | Streetwear / casual |
| `image4.jpg` | Formal blazer + trousers |
| `image5.jpg` | Sportswear / athleisure |

### 4. Edit the catalogue

Open `data/local_dataset_adapter.py` and update the `CATALOGUE` list at the top to match your actual image filenames and describe what's in each photo:

```python
CATALOGUE = [
    {
        "filename": "image1.jpg",
        "items": [
            {"Item Name": "Floral Wrap Midi Dress", "Price": "89.99", "Link": "https://..."},
            {"Item Name": "Strappy Block Heel Sandals", "Price": "59.99", "Link": "https://..."},
        ],
    },
    # ... one entry per image
]
```

### 5. Build the vector index

```bash
python ingest.py
```

This embeds all your images with CLIP and saves a FAISS index to `data/faiss_index/`. Takes about 10–30 seconds for 5 images.

### 6. Start the app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Usage

1. Open the app in your browser
2. Upload a fashion photo using the image panel on the left
3. Click **Analyse Style ✨**
4. The right panel shows a structured Markdown report with:
   - **Garments** — each visible clothing item with colour, pattern, material
   - **Accessories** — bags, shoes, jewellery, etc.
   - **Overall Style** — style category (e.g. business casual, streetwear)
   - **Matched Catalogue Items** — the retrieved products with prices and links

Your catalogue images also appear as clickable examples at the bottom of the UI.

---

## Configuration

All settings live in `.env`. Defaults work out of the box — only `GROQ_API_KEY` is required.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Get free at console.groq.com |
| `GROQ_MODEL_ID` | `Llama 4 Scout 17B 16E` | Must be a vision-capable Groq model |
| `GROQ_TEMPERATURE` | `0.2` | Lower = more deterministic output |
| `GROQ_MAX_TOKENS` | `2048` | Max response length |
| `EMBEDDING_MODEL_NAME` | `clip-ViT-B-32` | CLIP variant (512-dim, fast) |
| `FAISS_INDEX_PATH` | `data/faiss_index` | Where the index is saved |
| `IMAGES_DIR` | `data/images` | Folder containing your images |
| `TOP_K_RESULTS` | `5` | Number of similar items retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.75` | Cosine similarity cutoff for "exact match" language |

---

## Rebuilding the Index

After adding, removing, or changing images, rebuild the index with:

```bash
python ingest.py --force
```

To use a different images folder:

```bash
python ingest.py --images-dir path/to/my/images
```

---

## Tech Stack

| Component | Library | Role |
|---|---|---|
| Image embeddings | `sentence-transformers` (CLIP ViT-B/32) | Convert images to semantic vectors |
| Vector search | `faiss-cpu` | Fast cosine similarity search |
| LLM | `langchain-groq` + Groq API | Vision language model inference |
| RAG pipeline | `langchain` (LCEL) | Composable retrieval + generation chain |
| Config | `pydantic-settings` | Type-safe environment variable management |
| UI | `gradio` | Web interface |

---

## Extending the Project

**Swap to the HuggingFace dataset (44k items)** — Replace `local_dataset_adapter.py` with `dataset_adapter.py` and run `python ingest.py --samples 2000`. Better match quality at the cost of a longer ingestion time.

**Add a better CLIP model** — Change `EMBEDDING_MODEL_NAME=clip-ViT-L-14` in `.env` for 768-dim embeddings with higher accuracy (but slower inference). Update `FAISS_INDEX_PATH` to a new directory and re-run `ingest.py`.

**Enable LangSmith tracing** — Add to `.env`:
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
```
Every chain invocation is then visible in the LangSmith dashboard with full step-by-step traces.

**GPU acceleration** — Swap `faiss-cpu` for `faiss-gpu` in `requirements.txt`. CLIP inference also automatically uses CUDA if available.

---

## License

MIT