import time
import requests
from .config import (
    MODEL_BACKEND,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


def call_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 64):
    """
    Call LLM based on configured backend.

    Args:
        prompt: The prompt text
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum tokens to generate

    Returns:
        tuple: (response_text, metadata_dict)
    """
    start = time.time()
    backend = MODEL_BACKEND.lower()

    if backend == "ollama":
        return call_ollama(prompt, temperature, max_tokens, start)
    elif backend == "openai":
        return call_openai(prompt, temperature, max_tokens, start)
    else:
        # Should never happen due to config.py validation
        raise ValueError(f"Unknown backend: {backend}")


def call_ollama(prompt: str, temperature: float, max_tokens: int, start_time: float):
    """
    Call Ollama API.

    Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
    """
    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        text = data.get("response", "").strip()

        # Build metadata
        meta = {
            "latency_s": time.time() - start_time,
            "model": OLLAMA_MODEL,
            "backend": "ollama",
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "total_tokens": (
                data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            ),
        }

        return text, meta

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            f"Is Ollama running? Start with: ollama run {OLLAMA_MODEL}"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            f"Ollama request timed out. The model might be loading. "
            f"Try running: ollama run {OLLAMA_MODEL}"
        )
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}")


def call_openai(prompt: str, temperature: float, max_tokens: int, start_time: float):
    """
    Call OpenAI API.
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        text = response.choices[0].message.content.strip()

        # Build metadata
        usage = response.usage
        meta = {
            "latency_s": time.time() - start_time,
            "model": OPENAI_MODEL,
            "backend": "openai",
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

        return text, meta

    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install openai")
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}")
