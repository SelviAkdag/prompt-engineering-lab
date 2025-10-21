import pathlib
import json
import sys
import os
import io

# Fix Windows console encoding for emojis
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add project root to Python path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file BEFORE importing src modules
from dotenv import load_dotenv

load_dotenv()

import requests
from src.progress import load_progress, write_receipt

DATASET = pathlib.Path("data/sentiment_tiny.csv")


def fail(msg, metrics=None):
    print("FAIL:", msg)
    write_receipt("FAIL", metrics or {})
    sys.exit(1)


def validate_llm_backend():
    """
    Validate that LLM backend is properly configured.
    Called at the start of verify.py to catch config issues early.
    """
    backend = os.getenv("MODEL_BACKEND", "").lower()

    # Check that backend is set
    if not backend:
        return False, (
            "MODEL_BACKEND not set in .env file.\n"
            "   Set MODEL_BACKEND=ollama (recommended) or MODEL_BACKEND=openai"
        )

    # Check valid backend
    if backend not in ["ollama", "openai"]:
        return False, (
            f"Invalid MODEL_BACKEND: '{backend}'\n   Must be 'ollama' or 'openai'"
        )

    # Validate Ollama setup
    if backend == "ollama":
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code != 200:
                return False, (
                    "Ollama API returned error. Is Ollama running?\n"
                    "   Start with: ollama serve"
                )

            # Check if llama3.2:3b is available
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            if "llama3.2:3b" not in models:
                return False, (
                    "Model llama3.2:3b not found in Ollama.\n"
                    "   Download with: ollama pull llama3.2:3b"
                )

            return True, "✅ Ollama backend validated successfully"

        except requests.exceptions.ConnectionError:
            return False, (
                "Cannot connect to Ollama at http://localhost:11434\n"
                "   Is Ollama installed and running?\n"
                "   Install from: https://ollama.ai/download\n"
                "   Start with: ollama serve"
            )
        except Exception as e:
            return False, f"Error checking Ollama: {e}"

    # Validate OpenAI setup
    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return False, (
                "MODEL_BACKEND is 'openai' but OPENAI_API_KEY is not set.\n"
                "   Add your API key to .env file, or switch to MODEL_BACKEND=ollama"
            )

        if not api_key.startswith("sk-"):
            return False, (
                "OPENAI_API_KEY looks invalid (should start with 'sk-').\n"
                "   Check your API key at: https://platform.openai.com/api-keys"
            )

        return True, "✅ OpenAI backend validated successfully"

    return False, "Unknown error in backend validation"


def main():
    # Validate LLM backend first
    print("=" * 60)
    print("🔍 Validating LLM backend configuration...")
    print("=" * 60)

    valid, message = validate_llm_backend()
    if not valid:
        print("\n❌ LLM Backend validation failed:\n")
        print(f"   {message}\n")
        fail("LLM backend not properly configured")

    print(message)
    print()

    # Check dataset exists
    if not DATASET.exists():
        fail("Dataset missing.")

    # Load progress
    prog = load_progress()
    metrics = prog.get("metrics", {})
    quiz = prog.get("quiz", {})

    # Check all required methods have metrics
    required = ["zero_shot", "few_shot", "cot", "self_consistency"]
    for r in required:
        if r not in metrics:
            fail(f"Missing metrics for section: {r}")
        for m in ["accuracy", "precision", "recall", "f1"]:
            if m not in metrics[r]:
                fail(f"Metric {m} missing for {r}", metrics)

    # Check that at least one method reached 60% accuracy
    if not any(metrics[s].get("accuracy", 0) >= 0.60 for s in required):
        fail("No method reached accuracy >= 0.60", metrics)

    # Check quiz questions (one per section)
    expected_quizzes = [
        "quiz_zero_shot_q1",
        "quiz_few_shot_q1",
        "quiz_cot_q1",
        "quiz_self_consistency_q1",
    ]

    for q in expected_quizzes:
        if q not in quiz or not quiz[q]:
            fail(f"Quiz/reflection missing: {q}", metrics)

    # Check reflection questions
    required_reflections = [
        "reflection_best_method",
        "reflection_trade_offs",
        "reflection_real_world",
    ]

    for r in required_reflections:
        if r not in quiz or not quiz[r]:
            fail(
                f"Reflection missing: {r}. Please complete the comparison reflection section.",
                metrics,
            )

    # All checks passed!
    print("=" * 60)
    print("✅ PASS: All checks passed!")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   ✅ All 4 methods completed with metrics")
    print(f"   ✅ All {len(expected_quizzes)} quiz questions answered")
    print(f"   ✅ All {len(required_reflections)} reflections completed")
    print(f"   ✅ At least one method reached 60% accuracy")
    print("\n🎉 Congratulations! You have completed the Prompt Engineering Lab.")
    print("📄 Receipt saved to: progress/receipt.json\n")

    write_receipt("PASS", metrics)


if __name__ == "__main__":
    main()
