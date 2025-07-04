from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Nom de l'application
    APP_NAME: str = "RAG Juridique Camerounais"
    API_PREFIX: str = "/api"
    DEBUG: bool = True
    
    # Chemins de données
    DATA_PATH: str = "/Users/imacpro/modelAi/data"
    METADATA_PATH: str = "/Users/imacpro/modelAi/metadata"
    
    # Configuration Milvus
    MILVUS_HOST: str = "10.100.212.133"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "documents_collection"
    EMBEDDING_DIM: int = 1024
    
    # Configuration du service d'embedding
    EMBEDDING_SERVICE_URL: str = "http://192.168.50.215:8000"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"  # Ajouté ce champ manquant 192.168.50.215 10.100.210.10
    
    # Configuration LLM
    LLM_SERVICE_URL: str = "http://192.168.50.215:8001/generate"
    LLM_MODEL: str = "llama3.2:latest"
    
    # Configuration OCR
    OCR_LANGUAGE: str = "fra"
    FORCE_OCR: bool = False
    
    # Configuration de recherche
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 100
    MONOT5_MODEL: str = "castorini/monot5-base-msmarco"  
    
    # Configuration de chunking
    MAX_TOKENS: int = 512
    OVERLAP_TOKENS: int = 100

    # Paramètres LLM par défaut
    LLM_MAX_LENGTH: int = 500
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 0.9
    LLM_TOP_K: int = 50
       
    class Config:
        env_file = ".env"
        case_sensitive = True

# Instancier les paramètres
settings = Settings()

def get_settings() -> Settings:
    return settings