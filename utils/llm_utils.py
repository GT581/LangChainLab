from langchain_google_genai import ChatGoogleGenerativeAI

from utils.config import get_settings


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Get a Gemini LLM instance
        
    Returns:
        ChatGoogleGenerativeAI: Configured Gemini LLM instance
    """
    settings = get_settings()
    
    return ChatGoogleGenerativeAI(
        model=settings.GOOGLE_MODEL_NAME,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        top_p=settings.LLM_TOP_P,
        max_output_tokens=settings.LLM_MAX_TOKENS,
    )