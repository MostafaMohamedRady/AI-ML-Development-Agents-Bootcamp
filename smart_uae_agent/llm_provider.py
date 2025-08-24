# llm_provider.py
"""
Helpers to load the LLM and embeddings based on environment variables.

Env:
- PROVIDER: "gemini" (default) or "openai"
- GEMINI_MODEL: default "gemini-1.5-flash"
- OPENAI_MODEL: default "gpt-4o-mini"
- GOOGLE_API_KEY / GENAI_API_KEY: Gemini key
- OPENAI_API_KEY: OpenAI key
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from zope.interface import provider

load_dotenv()

def load_llm():
    """Return a LangChain chat model based on PROVIDER env var."""
    provider = os.getenv("PROVIDER", "gemini").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0.3)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=0.3)
    else:
        raise RuntimeError(f"Unknown provider {provider}")

def load_embeddings():
    """Return an embedding model for optional vector search (not required)."""
    provider = os.getenv("PROVIDER", "gemini").lower()
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/embeddings-001")
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        raise RuntimeError(f"Unknown provider for embeddings '{provider}'.")
