from .llm import call_llm


def build_zero_shot_prompt(text):
    """
    Build a zero-shot prompt for sentiment classification.

    Args:
        text: Input text to classify

    Returns:
        Formatted prompt string
    """
    return (
        f"You are a sentiment classifier. Respond with exactly one word: 'positive' or 'negative'.\n"
        f"Text: {text}\n"
        f"Answer:"
    )


def build_few_shot_prompt(text, demos=None):
    """
    Build a few-shot prompt with example demonstrations.

    Args:
        text: Input text to classify
        demos: List of (text, label) tuples. If None, uses default examples.

    Returns:
        Formatted prompt string with examples
    """
    if demos is None:
        demos = [
            ("I loved the film, it was wonderful!", "positive"),
            ("The acting was bad and the script was messy.", "negative"),
            ("A charming and uplifting story.", "positive"),
        ]

    demo_str = "\n".join([f"Text: {t}\nAnswer: {y}" for t, y in demos])

    return (
        f"You are a sentiment classifier. Respond with exactly one word: 'positive' or 'negative'.\n"
        f"{demo_str}\n"
        f"Text: {text}\n"
        f"Answer:"
    )


def build_cot_prompt(text):
    """
    Build a Chain-of-Thought prompt that asks for reasoning before the answer.

    Args:
        text: Input text to classify

    Returns:
        Formatted prompt string requesting step-by-step reasoning
    """
    return (
        f"You are a helpful reasoning assistant. Analyze the sentiment step by step.\n\n"
        f"Text: {text}\n\n"
        f"Instructions:\n"
        f"1. Show your reasoning step by step\n"
        f"2. On the LAST line, write ONLY: 'Answer: positive' OR 'Answer: negative'\n\n"
        f"Reasoning:"
    )


def normalize_label(s):
    """
    Normalize model output to 'positive' or 'negative'.

    Args:
        s: Raw model output string

    Returns:
        'positive' or 'negative'
    """
    if not s:
        return "negative"  # Default fallback

    s_clean = s.strip().lower()

    # Direct matches (highest priority)
    if s_clean == "positive":
        return "positive"
    if s_clean == "negative":
        return "negative"

    # Substring matches
    if "positive" in s_clean or "pos" in s_clean:
        return "positive"
    if "negative" in s_clean or "neg" in s_clean:
        return "negative"

    # Keyword-based heuristic (last resort)
    positive_words = {"good", "great", "excellent", "wonderful", "love", "amazing"}
    negative_words = {"bad", "terrible", "poor", "hate", "awful", "worst"}

    words = set(s_clean.split())

    pos_count = len(words & positive_words)
    neg_count = len(words & negative_words)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"

    # Final fallback
    return "negative"


def run_zero_shot(dataset):
    """
    Run zero-shot classification on a dataset.

    Args:
        dataset: List of (text, label) tuples

    Returns:
        preds: List of predicted labels
        metas: List of metadata dicts from each call
    """
    preds, metas = [], []

    for text, _ in dataset:
        try:
            p = build_zero_shot_prompt(text)
            ans, meta = call_llm(p, temperature=0.0, max_tokens=5)
            preds.append(normalize_label(ans))
            metas.append(meta)
        except Exception as e:
            print(f"⚠️ Error processing text: {text[:50]}... | Error: {e}")
            preds.append("negative")  # Fallback
            metas.append({"error": str(e)})

    return preds, metas


def run_few_shot(dataset, demos=None):
    """
    Run few-shot classification on a dataset.

    Args:
        dataset: List of (text, label) tuples
        demos: Optional list of (text, label) example tuples

    Returns:
        preds: List of predicted labels
        metas: List of metadata dicts from each call
    """
    preds, metas = [], []

    for text, _ in dataset:
        try:
            p = build_few_shot_prompt(text, demos=demos)
            ans, meta = call_llm(p, temperature=0.0, max_tokens=5)
            preds.append(normalize_label(ans))
            metas.append(meta)
        except Exception as e:
            print(f"⚠️ Error processing text: {text[:50]}... | Error: {e}")
            preds.append("negative")
            metas.append({"error": str(e)})

    return preds, metas


def run_cot(dataset):
    """
    Run Chain-of-Thought classification on a dataset.
    """
    preds, metas = [], []

    for text, _ in dataset:
        try:
            p = build_cot_prompt(text)
            ans, meta = call_llm(p, temperature=0.0, max_tokens=300)

            # 🔧 FÖRBÄTTRAD PARSING: Leta efter "Answer:" pattern
            if ans and "answer:" in ans.lower():
                # Hitta sista raden med "answer:"
                lines = ans.strip().lower().split("\n")
                for line in reversed(lines):
                    if "answer:" in line:
                        # Extrahera efter "answer:"
                        answer_part = line.split("answer:")[-1].strip()
                        preds.append(normalize_label(answer_part))
                        break
                else:
                    # Om loop slutar utan break
                    preds.append("negative")
            else:
                # Fallback: använd sista ordet (old behavior)
                last_word = ans.strip().split()[-1] if ans and ans.strip() else ""
                preds.append(normalize_label(last_word))

            metas.append(meta)
        except Exception as e:
            print(f"⚠️ Error processing text: {text[:50]}... | Error: {e}")
            preds.append("negative")
            metas.append({"error": str(e)})

    return preds, metas


def run_self_consistency(dataset, k=5):
    """
    Run Self-Consistency (multiple CoT samples + majority vote).
    """
    import collections

    preds, metas = [], []

    for text, _ in dataset:
        votes = []
        local_metas = []

        for _ in range(k):
            try:
                p = build_cot_prompt(text)
                ans, meta = call_llm(p, temperature=0.7, max_tokens=300)

                # 🔧 FÖRBÄTTRAD PARSING: Samma som run_cot()
                if ans and "answer:" in ans.lower():
                    lines = ans.strip().lower().split("\n")
                    for line in reversed(lines):
                        if "answer:" in line:
                            answer_part = line.split("answer:")[-1].strip()
                            votes.append(normalize_label(answer_part))
                            break
                    else:
                        votes.append("negative")
                else:
                    # Fallback
                    last_word = ans.strip().split()[-1] if ans and ans.strip() else ""
                    votes.append(normalize_label(last_word))

                local_metas.append(meta)
            except Exception as e:
                print(f"⚠️ Error in self-consistency: {e}")
                votes.append("negative")
                local_metas.append({"error": str(e)})

        # Majority vote
        if votes:
            majority = collections.Counter(votes).most_common(1)[0][0]
            preds.append(majority)
        else:
            preds.append("negative")

        metas.extend(local_metas)

    return preds, metas
