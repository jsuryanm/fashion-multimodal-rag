import logging 
import pickle 
from pathlib import Path 

import faiss 
import numpy as np 

logger = logging.getLogger(__name__)

class FashionVectorStore:

    def __init__(self,dimension: int):
        self.dimension = dimension 
        self.index = faiss.IndexFlatIP(dimension) # dimesnion is the no of indexes
        self.metadata: list[dict] = []
    
    def add_items(self,
                  embeddings:np.array,
                  metadata_rows: list[dict]) -> None:
        """
        Add embeddings and their metadata to the store.
 
        Args:
            embeddings:     Float32 ndarray shape (N, dimension). Must be L2-normalised.
            metadata_rows:  List of N dicts. Each dict must have at minimum:
                            Image URL, Item Name, Price, Link.
                            all_items (list of dicts) is also stored for full outfit retrieval.
        """
        
        if len(embeddings) != len(metadata_rows):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(metadata_rows)} metadata rows")
        
        # add the vectors to faiss
        self.index.add(np.asarray(embeddings,dtype=np.float32))
        self.metadata.extend(metadata_rows)
        # create a mapping for faiss index -> metadata (cloths,price,etc)
        logger.info(f"Added {len(metadata_rows)} vectors. Total: {self.index.ntotal}")

    
    def search(self,query_vector: np.ndarray, top_k: int = 5) -> list[tuple[dict,float]]:
        """
        Find the top-k most visually similar itemm

        Returns:
            List of (metadata_dict,cosine_similarity_score) sorted by
            descending similarity (best match first)
        """

        query = np.asarray([query_vector],dtype=np.float32)
        distances,indices = self.index.search(query,top_k)
        # the output of faiss search is a 2d vector so iterate through the inner most dim

        results = []
        
        for dist,idx in zip(distances[0],indices[0]):
            if idx < 0:
                continue 

            results.append((self.metadata[idx],float(dist)))
        return results 
    
    def get_best_match(self,query_vector: np.ndarray) -> tuple[dict | None,float]:
        results = self.search(query_vector, top_k=1)
        return results[0] if results else (None, 0.0)
    
    def save(self,directory: str| Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        faiss.write_index(self.index,str(directory/"index.faiss"))
        with open(directory / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info("Vector store saved to '%s' (%d items)", directory, self.index.ntotal)
 
    @classmethod
    def load(cls, directory: str | Path) -> "FashionVectorStore":
        directory = Path(directory)
        index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        instance = cls.__new__(cls)
        instance.index = index
        instance.dimension = index.d
        instance.metadata = metadata
        logger.info("Vector store loaded from '%s' (%d items)", directory, index.ntotal)
        return instance
 
    @classmethod
    def exists(cls, directory: str | Path) -> bool:
        d = Path(directory)
        return (d / "index.faiss").exists() and (d / "metadata.pkl").exists()