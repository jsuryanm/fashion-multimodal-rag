from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import Field 

class Settings(BaseSettings):
    
    model_config = SettingsConfigDict(env_file=".env",
                                      env_file_encoding="utf-8",
                                      case_sensitive=False)
    
    groq_api_key: str = Field(...,description="Groq API key")
    groq_model_id: str = Field(default="meta-llama/llama-4-scout-17b-16e-instruct",
                               description="Groq model identifier")
    groq_temperature: float = Field(default=0.2,ge=0.1,le=1)
    groq_max_tokens: int = Field(default=2048,gt=0)


    embeddings_model_name: str = Field(default="clip-ViT-B-32",
                                  description="SentenceTransformers model name for image embeddings")
    
    faiss_index_path: str = Field(default="data/faiss_index",description="Local directory to persist the FAISS index")
    images_dir: str = Field(default="data")
    
    top_k_results: int = Field(default=5,gt=0)
    similarity_threshold: float = Field(default=0.75,ge=0.0,le=1.0)

settings = Settings()