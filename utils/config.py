from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    LLM settings configuration
    """
    # LLM Config - API Key loaded from .env
    GOOGLE_API_KEY: str
    GOOGLE_MODEL_NAME: str = "gemini-1.5-flash"

    LLM_TEMPERATURE: float = 0.1
    LLM_TOP_P: float = 0.95
    LLM_MAX_TOKENS: int = 8192
    
    # Load API Key from .env file
    class Config:
        env_file = ".env"
        case_sensitive = True

def get_settings() -> Settings:
    """
    Get LLM settings with caching.
    
    Returns:
        Settings: LLM settings
    """
    return Settings() 