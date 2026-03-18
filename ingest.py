import argparse 
import logging 
from pathlib import Path

from src.config.settings import settings
from src.data_loader.local_dataset_adapter import load_local_dataset
from src.models.embeddings import CLIPImageEmbedder
from src.models.vector_store import FashionVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)
 
"""
argparse is way to pass inputs to Python script from terminal



"""

def parse_args() -> argparse.Namespace:
    """Read terminal arguments and return them as variables."""

    parser = argparse.ArgumentParser(description="Build FAISS index from local images")
    
    parser.add_argument("--force",
                        action="store_true",
                        help="Rebuild even if an index already exist")
    
    parser.add_argument("--images-dir",
                        default=settings.images_dir,
                        help=f"Folder with your images (default: {settings.images_dir})")
    
    return parser.parse_args()

def ingest(images_dir: str,force: bool = False) -> None:
    index_path = settings.faiss_index_path

    if FashionVectorStore.exists(index_path) and not force:
        logger.info(f"FAISS index already exists at {index_path}. Use --force to rebuild this")
        return
    
    # load images 
    logger.info(f"Loading images from {images_dir}")
    df = load_local_dataset(images_dir)

    if df.empty:
        logger.error(
            "No images loaded. Check that:\n"
            "  1. Your images are in '%s'\n"
            "  2. Filenames in CATALOGUE (local_dataset_adapter.py) match your files",
            images_dir,
        )
        return
    

    # De-duplicate: one CLIP embedding per unique image file
    unique_images_df = df.drop_duplicates(subset="Image URL")
    logger.info(
        "  -> %d unique images found, %d total catalogue items",
        len(unique_images_df),
        len(df),
    )

    # Compute CLIP embeddings 
    logger.info("Loading CLIP model and compute embeddings")
    logger.info("Downloading from hf hub")
    
    embedder = CLIPImageEmbedder(settings.embeddings_model_name)
    
    pil_images = unique_images_df["image"].tolist()
    
    embeddings = embedder.embed_images_batch(pil_images)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    dimension = embeddings.shape[1]
    store = FashionVectorStore(dimension=dimension)
    metadata_rows = []
    
    for _,row in unique_images_df.iterrows():
        image_url = row['Image URL']
        all_items = (
            df[df["Image URL"] == image_url][["Item Name", "Price", "Link"]]
            .to_dict("records")
        )
        metadata_rows.append(
            {
                "Image URL": image_url,
                "Item Name": row["Item Name"],   # primary item (first listed)
                "Price": row["Price"],
                "Link": row["Link"],
                "all_items": all_items,          # full outfit item list
            }
        )
 
    store.add_items(embeddings, metadata_rows)
    store.save(index_path)
 
    logger.info("")
    logger.info("Ingestion complete!")
    logger.info("Images indexed : %d", store.index.ntotal)
    logger.info("Index saved to : %s", index_path)
    logger.info("")
    logger.info("   Start the app  : python app.py")
 
 
if __name__ == "__main__":
    args = parse_args()
    ingest(images_dir=args.images_dir, force=args.force)
 