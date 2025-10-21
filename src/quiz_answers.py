# src/quiz_answers.py
"""
Quiz answers and feedback for the prompt engineering lesson.
Students should not look at this file! 🙈
"""

QUIZ_DATA = {
    "zero_shot_q1": {
        "correct_answer": "The model relies only on instructions, without examples.",
        "correct_feedback": """
            <strong style="color:#065f46;">✅ Correct!</strong> Zero-shot means no examples in the prompt.<br>
            <span style="color:#047857;">The model relies solely on its pre-trained knowledge.</span>
        """,
        "incorrect_feedback": """
            <strong style="color:#991b1b;">❌ Not quite!</strong> Think about what 'zero-shot' means.<br>
            <span style="color:#b91c1c;">💡 <em>Hint: How many examples does the model see?</em></span>
        """,
    },
    "few_shot_q1": {
        "correct_answer": "Providing labeled examples in the prompt to guide the model.",
        "correct_feedback": """
            <strong style="color:#065f46;">✅ Correct!</strong> Few-shot learning uses a few examples to guide the model.
        """,
        "incorrect_feedback": """
            <strong style="color:#991b1b;">❌ Try again!</strong> Think about how few-shot differs from zero-shot.<br>
            <span style="color:#b91c1c;">💡 <em>Hint: These demonstrations help the model understand the pattern and output format.</em></span> 
        """,
    },
    "cot_q1": {
        "correct_answer": "Ask the model to explain its reasoning step-by-step",
        "correct_feedback": """
            <strong style="color:#065f46;">✅ Correct!</strong> Chain-of-Thought prompting encourages explicit reasoning.
        """,
        "incorrect_feedback": """
            <strong style="color:#991b1b;">❌ Not quite!</strong> CoT focuses on the reasoning process, not just the answer.
        """,
    },
    "self_consistency_q1": {
        "correct_answer": "Run multiple reasoning paths and use majority vote",
        "correct_feedback": """
            <strong style="color:#065f46;">✅ Correct!</strong> Self-consistency improves reliability through sampling.
        """,
        "incorrect_feedback": """
            <strong style="color:#991b1b;">❌ Try again!</strong> Think about how multiple samples can improve accuracy.
        """,
    },
}


def check_answer(question_id, user_answer):
    """
    Check if user's answer is correct and return feedback.

    Args:
        question_id: Key from QUIZ_DATA dict
        user_answer: User's selected option

    Returns:
        tuple: (is_correct: bool, feedback_html: str)
    """
    if question_id not in QUIZ_DATA:
        return False, "Question not found."

    quiz_info = QUIZ_DATA[question_id]
    is_correct = user_answer == quiz_info["correct_answer"]

    if is_correct:
        feedback = f"""
        <div style="background:#d1fae5; border-left:4px solid #10b981; padding:14px 16px; border-radius:6px; margin-top:12px; color:#065f46;">
            {quiz_info["correct_feedback"]}
        </div>
        """
    else:
        feedback = f"""
        <div style="background:#fee2e2; border-left:4px solid #ef4444; padding:14px 16px; border-radius:6px; margin-top:12px; color:#991b1b;">
            {quiz_info["incorrect_feedback"]}
        </div>
        """

    return is_correct, feedback
