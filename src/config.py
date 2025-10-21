import os
from dotenv import load_dotenv

load_dotenv()

# Backend selection
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "ollama")

# Ollama configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Validate backend choice
if MODEL_BACKEND not in ["ollama", "openai"]:
    raise ValueError(
        f"Invalid MODEL_BACKEND: '{MODEL_BACKEND}'. "
        f"Must be 'ollama' or 'openai'. Check your .env file."
    )

# Validate OpenAI setup if selected
if MODEL_BACKEND == "openai" and not OPENAI_API_KEY:
    raise ValueError(
        "MODEL_BACKEND is 'openai' but OPENAI_API_KEY is not set. "
        "Add your API key to .env or switch to MODEL_BACKEND=ollama"
    )
