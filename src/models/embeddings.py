import logging 
import numpy as np 
from sentence_transformers import SentenceTransformer
from src.config.settings import settings
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPImageEmbedder:
    
    def __init__(self,model_name: str = settings.embeddings_model_name):
        
        logger.info(f"Loading CLIP embeddings model:{model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"CLIP Ready. Embedding dimension: {self.dimension}")
        
    def embed_image(self,image: Image.Image) -> np.ndarray:
        """
        Embed a single PIL image
        Returns:
        1D float32 np array of shape (dim,) 
        """
        embeddings = self.model.encode([image],
                                       convert_to_numpy=True,
                                       normalize_embeddings=True)
        return embeddings[0]
    
    def embed_images_batch(self,images: list[Image.Image]) -> np.ndarray:
        """
        Embed a list of PIL Images efficiently via batching
        Returns: 
        2-D float32 np array (n,dim)
        """

        logger.info(f"Embeddings {len(images)} images")
        embeddings = self.model.encode(images,
                                 convert_to_numpy=True,
                                 normalize_embeddings=True,
                                 show_progress_bar=True,
                                 batch_size=32)

        if self.dimension is None:
            self.dimension = embeddings.shape[1]
            logger.info(f"Inferred embedding dimension: {self.dimension}")
        
        return embeddings