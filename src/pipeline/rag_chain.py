import logging 
import numpy as np 
from PIL import Image 

from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from src.config.settings import settings 
from src.models.embeddings import CLIPImageEmbedder
from src.models.llm_service import build_fashion_messages,build_groq_llm
from src.models.vector_store import FashionVectorStore
from src.utils.image_utils import load_image,pil_to_data_uri
from src.utils.response_formatter import format_response


logger = logging.getLogger(__name__)

class FashionRAGChain:
    """Orchestrates full multimodal RAG Chain"""

    def __init__(self,
                 embedder: CLIPImageEmbedder,
                 vector_store: FashionVectorStore):
        
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = build_groq_llm(groq_api_key=settings.groq_api_key,
                                  model_id=settings.groq_model_id,
                                  temperature=settings.groq_temperature,
                                  max_tokens=settings.groq_max_tokens)
        
        
        self.chain = self._build_chain()

    
    def _build_chain(self): 
        rag_chain = (
            RunnableLambda(self._load_and_embed)
            | RunnableLambda(self._retrieve_context)
            | RunnableLambda(self._build_prompt)
            | self.llm 
            | StrOutputParser()
            | RunnableLambda(format_response)
        )

        return rag_chain 
    
    def _load_and_embed(self,input_data: dict) -> dict:
        """Load img from file path
        Compute CLIP Embeddings"""

        source = input_data['image_source']
        
        if isinstance(source,Image.Image):
            image = source.convert("RGB")
        else:
            image = load_image(source)

        logger.info(f"Embeddings query image:{image.size}")
        embedding  = self.embedder.embed_image(image)
        data_uri = pil_to_data_uri(image)

        return {**input_data, "image": image, "embedding": embedding, "data_uri": data_uri}


    def _retrieve_context(self,state: dict) -> dict:
        """Query FAISS for the top-k most visually similar catalogue items."""
        results = self.vector_store.search(state["embedding"],
                                           top_k=settings.top_k_results)
        
        if not results:
            logger.warning("FAISS returned no results")
            return {**state, "matched_items": [], "best_score": 0.0, "is_exact_match": False}
        
        matched_items = [item for item,_ in results]
        best_score = results[0][1]
        is_exact = best_score >= settings.similarity_threshold

        logger.info(
            "Best match: '%s' | score=%.3f | exact=%s",
            matched_items[0].get("Item Name", "?"),
            best_score,
            is_exact,
        )

        return {**state,
                "matched_items":matched_items,
                "best_score":best_score,
                "is_exact_match":is_exact}
    
    def _build_prompt(self, state: dict) -> list:
        """
        Build the LangChain message list.
 
        If nothing was retrieved (empty index or embedding failure),
        fall back to a simple describe-what-you-see prompt so the
        pipeline degrades gracefully rather than crashing.
        """
        if not state["matched_items"]:
            logger.warning("No retrieved items — using fallback prompt")
            return [
                SystemMessage(content="You are a professional fashion analyst."),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe the clothing and style in this image."},
                        {"type": "image_url", "image_url": {"url": state["data_uri"]}},
                    ]
                ),
            ]
 
        return build_fashion_messages(
            image_data_uri=state["data_uri"],
            matched_items=state["matched_items"],
            is_exact_match=state["is_exact_match"],
        )
    
    def invoke(self, image_source) -> str:
        """
        Run the pipeline synchronously.
 
        Args:
            image_source: file path (str/Path), URL (str), or PIL Image.
 
        Returns:
            Markdown-formatted fashion analysis string.
        """
        return self.chain.invoke({"image_source": image_source})
 
    async def ainvoke(self, image_source) -> str:
        """Async version — use in async Gradio event handlers."""
        return await self.chain.ainvoke({"image_source": image_source})