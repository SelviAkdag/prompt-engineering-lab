"""
Helper functions for interactive notebook lesson.
Provides display, quiz, and execution functions for all prompting methods.
"""

import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from .prompting import (
    build_zero_shot_prompt,
    build_few_shot_prompt,
    build_cot_prompt,
    run_zero_shot,
    run_few_shot,
    run_cot,
    run_self_consistency,
    normalize_label,
)
from .llm import call_llm
from .metrics import precision_recall_f1
from .progress import load_progress, save_progress
from .quiz_answers import check_answer


# ============================================================================
# SECTION 1: ZERO-SHOT HELPERS
# ============================================================================


def display_zero_shot_example(rows):
    """
    Display a complete zero-shot classification example with styling.

    Args:
        rows: Dataset as list of (text, label) tuples
    """
    print("🎬 Let's see how zero-shot classification works!\n")

    # Step 1: Choose an example
    sentence = "The movie was absolutely wonderful and full of emotion."
    expected = "positive"

    print(f"📄 Input sentence:")
    print(f'   "{sentence}"\n')
    print(f"🎯 Expected sentiment: {expected.upper()}\n")

    # Step 2: Build the prompt
    prompt = build_zero_shot_prompt(sentence)

    print("=" * 70)
    print("🔍 PROMPT SENT TO MODEL:")
    print("=" * 70)
    print(prompt)
    print("=" * 70)

    # Step 3: Call the model
    print("\n⏳ Calling model...\n")
    model_output, meta = call_llm(prompt, temperature=0.0, max_tokens=16)

    # Step 4: Normalize the output
    prediction = normalize_label(model_output)

    print("=" * 70)
    print("🤖 MODEL RESPONSE:")
    print("=" * 70)
    print(f"   Raw output: {model_output}")
    print(f"   Normalized: {prediction.upper()}")
    print(f"   Latency: {meta.get('latency_s', 0):.3f}s")
    print("=" * 70)

    # Step 5: Display results in a styled table
    print("\n📊 Results Summary:\n")

    data = {
        "Sentence": [sentence],
        "Model Prediction": [prediction.capitalize()],
        "Expected": [expected.capitalize()],
        "Result": [
            "✅ Correct" if prediction.lower() == expected.lower() else "❌ Incorrect"
        ],
    }

    df = pd.DataFrame(data)
    styled_df = df.style.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#3b82f6"),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("padding", "10px"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "left"),
                    ("padding", "8px"),
                    ("border-bottom", "1px solid #ddd"),
                ],
            },
        ]
    ).hide(axis="index")

    display(styled_df)

    # Step 6: Explain what happened
    print()
    if prediction.lower() == expected.lower():
        print("✅ Success! The model correctly identified the sentiment.")
    else:
        print("❌ The model made an error. This can happen in zero-shot settings.")
        print("   💡 Tip: Few-shot or Chain-of-Thought prompting might help!")


def create_zero_shot_interactive():
    """
    Interactive test - uses standard input() for consistent styling.
    """
    print("🎯 Test zero-shot classification with your own sentence!\n")

    sentence = input("Enter a sentence to classify: ")

    if sentence.strip():
        # Build prompt
        prompt = build_zero_shot_prompt(sentence)

        print(f"\n{'=' * 70}")
        print("🔍 PROMPT SENT TO MODEL:")
        print(f"{'=' * 70}")
        print(prompt)
        print(f"{'=' * 70}")

        # Call model
        print("\n⏳ Calling model...\n")
        model_output, meta = call_llm(prompt, temperature=0.0, max_tokens=16)
        prediction = normalize_label(model_output)

        print(f"🤖 Model prediction: {prediction.upper()}")
        print(f"⏱️  Latency: {meta.get('latency_s', 0):.3f}s")

        print("\n💡 The model relies only on its pre-trained knowledge!")
    else:
        print("❌ No input provided")


def create_zero_shot_quiz(progress):
    """
    Create and display the zero-shot quiz widget.

    Args:
        progress: Progress dictionary to save quiz results
    """
    # Display quiz header
    display(
        HTML("""
    <div style="background:#dbeafe; border-left:5px solid #2563eb; padding:14px 18px; margin:16px 0; border-radius:6px; color:#1e3a8a;">
        <strong style="font-size:16px;">🔍 Knowledge Check</strong><br>
        <span style="font-size:14px;">Test your understanding of zero-shot prompting before moving forward.</span>
    </div>
    """)
    )

    # Question label
    question_label = widgets.HTML(
        value="<b style='font-size:15px; color:#1f2937;'>❓ What defines zero-shot prompting?</b>",
        layout=widgets.Layout(margin="10px 0 8px 0"),
    )

    # Radio buttons
    q1 = widgets.RadioButtons(
        options=[
            "The model is given several examples before making a prediction.",
            "The model relies only on instructions, without examples.",
            "The model uses labeled datasets to learn during training.",
        ],
        value=None,
        description="",
        style={"description_width": "0px"},
        layout=widgets.Layout(width="100%", margin="10px 0"),
    )

    # Apply custom CSS
    display(
        HTML("""
    <style>
    .widget-radio-box label {
        color: #1f2937 !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
    }
    .widget-radio-box input[type="radio"] {
        margin-right: 8px !important;
    }
    </style>
    """)
    )

    output1 = widgets.Output()

    def check_q1(change):
        if change["new"] is None:
            return

        with output1:
            clear_output()
            is_correct, feedback_html = check_answer("zero_shot_q1", change["new"])
            display(HTML(feedback_html))

            if is_correct:
                quiz = progress.setdefault("quiz", {})
                quiz["quiz_zero_shot_q1"] = "correct"
                save_progress(progress)

    q1.observe(check_q1, names="value")

    display(question_label, q1, output1)


def run_zero_shot_full_dataset(rows, progress):
    """
    Run zero-shot classification on full dataset and display results.

    Args:
        rows: Dataset as list of (text, label) tuples
        progress: Progress dictionary to save metrics

    Returns:
        metrics_z: Dictionary with performance metrics
    """
    print("=" * 70)
    print(f"🚀 RUNNING ZERO-SHOT ON ALL {len(rows)} EXAMPLES")
    print("=" * 70)
    print("\nThis will classify all movie reviews using zero-shot prompting...")
    print("⏳ Please wait...\n")

    # Run zero-shot on entire dataset
    pred_z, metas_z = run_zero_shot(rows)
    true = [y for _, y in rows]
    metrics_z = precision_recall_f1(true, pred_z)
    metrics_z.update({"calls": len(pred_z)})

    # Save to progress
    progress.setdefault("metrics", {})["zero_shot"] = metrics_z
    save_progress(progress)

    print("✅ Zero-shot classification complete!\n")

    # Display metrics in a table
    print("📊 RESULTS SUMMARY:\n")

    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Score": [
            f"{metrics_z['accuracy']:.1%}",
            f"{metrics_z['precision']:.1%}",
            f"{metrics_z['recall']:.1%}",
            f"{metrics_z['f1']:.1%}",
        ],
        "Description": [
            "Overall correct predictions",
            "Positive precision",
            "Positive recall",
            "Harmonic mean",
        ],
    }

    df_metrics = pd.DataFrame(metrics_data)
    display(
        df_metrics.style.set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#3b82f6"),
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                        ("padding", "10px"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "left"),
                        ("padding", "8px"),
                        ("border-bottom", "1px solid #ddd"),
                    ],
                },
            ]
        ).hide(axis="index")
    )

    # Visualize metrics
    print("\n" + "=" * 70)
    print("📈 VISUALIZATION")
    print("=" * 70 + "\n")

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    metrics_values = [
        metrics_z["accuracy"],
        metrics_z["precision"],
        metrics_z["recall"],
        metrics_z["f1"],
    ]

    bars = ax.bar(
        metrics_names,
        metrics_values,
        color="#3b82f6",
        alpha=0.8,
        edgecolor="#1e40af",
        linewidth=2,
    )
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title(
        "Zero-shot Performance on Full Dataset", fontsize=15, fontweight="bold", pad=20
    )

    # Target line
    ax.axhline(
        y=0.6,
        color="#10b981",
        linestyle="--",
        linewidth=2.5,
        label="Target: 60%",
        alpha=0.8,
    )
    ax.legend(fontsize=11, loc="lower right")

    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=1)
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.03,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
    plt.tight_layout()
    plt.show()

    # Detailed breakdown
    print("\n" + "=" * 70)
    print("📋 DETAILED BREAKDOWN")
    print("=" * 70)
    print(f"\n   ✅ True Positives: {metrics_z['tp']}")
    print(f"   ✅ True Negatives: {metrics_z['tn']}")
    print(f"   ❌ False Positives: {metrics_z['fp']}")
    print(f"   ❌ False Negatives: {metrics_z['fn']}")
    print(f"   📞 Total API calls: {metrics_z['calls']}\n")

    # Check if target reached
    print("=" * 70)
    if metrics_z["accuracy"] >= 0.6:
        print("🎉 SUCCESS! Zero-shot reached the target accuracy of 60%!")
    else:
        print("⚠️  Zero-shot accuracy is below 60%.")
        print("💡 Tip: Few-shot prompting might improve performance.")
    print("=" * 70)

    return metrics_z


# ============================================================================
# SECTION 2: FEW-SHOT HELPERS
# ============================================================================


def display_few_shot_example(rows):
    """
    Display a complete few-shot classification example with styling.

    Args:
        rows: Dataset as list of (text, label) tuples
    """
    print("🎬 Let's see how few-shot classification works!\n")

    # Step 1: Choose an example
    sentence = "The movie was absolutely wonderful and full of emotion."
    expected = "positive"

    print(f"📄 Input sentence:")
    print(f'   "{sentence}"\n')
    print(f"🎯 Expected sentiment: {expected.upper()}\n")

    # Step 2: Define few-shot demonstrations
    demonstrations = [
        ("I loved the film, it was wonderful!", "positive"),
        ("The acting was bad and the script was messy.", "negative"),
        ("A charming and uplifting story.", "positive"),
    ]

    print("📚 Example demonstrations provided to the model:\n")
    for i, (demo_text, demo_label) in enumerate(demonstrations, 1):
        print(f'   {i}. "{demo_text}" → {demo_label}')
    print()

    # Step 3: Build the prompt with examples
    prompt = build_few_shot_prompt(sentence, demos=demonstrations)

    print("=" * 70)
    print("🔍 PROMPT SENT TO MODEL:")
    print("=" * 70)
    print(prompt)
    print("=" * 70)

    # Step 4: Call the model
    print("\n⏳ Calling model...\n")
    model_output, meta = call_llm(prompt, temperature=0.0, max_tokens=16)

    # Step 5: Normalize the output
    prediction = normalize_label(model_output)

    print("=" * 70)
    print("🤖 MODEL RESPONSE:")
    print("=" * 70)
    print(f"   Raw output: {model_output}")
    print(f"   Normalized: {prediction.upper()}")
    print(f"   Latency: {meta.get('latency_s', 0):.3f}s")
    print("=" * 70)

    # Step 6: Display results in a styled table
    print("\n📊 Results Summary:\n")

    data = {
        "Sentence": [sentence],
        "Model Prediction": [prediction.capitalize()],
        "Expected": [expected.capitalize()],
        "Result": [
            "✅ Correct" if prediction.lower() == expected.lower() else "❌ Incorrect"
        ],
    }

    df = pd.DataFrame(data)
    styled_df = df.style.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#8b5cf6"),  # Purple for few-shot
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("padding", "10px"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "left"),
                    ("padding", "8px"),
                    ("border-bottom", "1px solid #ddd"),
                ],
            },
        ]
    ).hide(axis="index")

    display(styled_df)

    # Step 7: Explain what happened
    print()
    if prediction.lower() == expected.lower():
        print(
            "✅ Success! The model correctly identified the sentiment using the examples."
        )
        print("   💡 The demonstrations helped guide the model's understanding.")
    else:
        print("❌ The model made an error despite having examples.")
        print("   💡 Tip: Try different examples or use Chain-of-Thought prompting!")


def create_few_shot_interactive():
    """
    Interactive test - uses standard input() for consistent styling.
    """
    print("🎯 Test few-shot classification with your own sentence!\n")

    sentence = input("Enter a sentence to classify: ")

    if sentence.strip():
        # Define demonstrations
        demonstrations = [
            ("I loved the film, it was wonderful!", "positive"),
            ("The acting was bad and the script was messy.", "negative"),
            ("A charming and uplifting story.", "positive"),
        ]

        print(f"\n📚 Using these demonstrations:")
        for i, (demo_text, demo_label) in enumerate(demonstrations, 1):
            print(f'   {i}. "{demo_text}" → {demo_label}')

        # Build prompt
        prompt = build_few_shot_prompt(sentence, demos=demonstrations)

        print(f"\n{'=' * 70}")
        print("🔍 PROMPT SENT TO MODEL:")
        print(f"{'=' * 70}")
        print(prompt)
        print(f"{'=' * 70}")

        # Call model
        print("\n⏳ Calling model...\n")
        model_output, meta = call_llm(prompt, temperature=0.0, max_tokens=16)
        prediction = normalize_label(model_output)

        print(f"🤖 Model prediction: {prediction.upper()}")
        print(f"⏱️  Latency: {meta.get('latency_s', 0):.3f}s")

        print("\n💡 Notice how the model uses the examples to understand the pattern!")
    else:
        print("❌ No input provided")


def create_few_shot_quiz(progress):
    """
    Create and display the few-shot quiz widget.

    Args:
        progress: Progress dictionary to save quiz results
    """
    # Display quiz header
    display(
        HTML("""
    <div style="background:#dbeafe; border-left:5px solid #2563eb; padding:14px 18px; margin:16px 0; border-radius:6px; color:#1e3a8a;">
        <strong style="font-size:16px;">🔍 Knowledge Check</strong><br>
        <span style="font-size:14px;">Test your understanding of few-shot prompting before moving forward.</span>
    </div>
    """)
    )

    # Question label
    question_label = widgets.HTML(
        value="<b style='font-size:15px; color:#1f2937;'>❓ What is the main characteristic of few-shot prompting?</b>",
        layout=widgets.Layout(margin="10px 0 8px 0"),
    )

    # Radio buttons
    q1 = widgets.RadioButtons(
        options=[
            "Providing labeled examples in the prompt to guide the model.",
            "The model is trained on thousands of labeled examples.",
            "The model generates multiple answers and picks the best one.",
        ],
        value=None,
        description="",
        style={"description_width": "0px"},
        layout=widgets.Layout(width="100%", margin="10px 0"),
    )

    # Apply custom CSS
    display(
        HTML("""
    <style>
    .widget-radio-box label {
        color: #1f2937 !important;
        font-size: 14px !important;
        line-height: 1.8 !important;
    }
    .widget-radio-box input[type="radio"] {
        margin-right: 10px !important;
    }
    </style>
    """)
    )

    output1 = widgets.Output()

    def check_q1(change):
        if change["new"] is None:
            return

        with output1:
            clear_output()
            is_correct, feedback_html = check_answer("few_shot_q1", change["new"])
            display(HTML(feedback_html))

            if is_correct:
                quiz = progress.setdefault("quiz", {})
                quiz["quiz_few_shot_q1"] = "correct"
                save_progress(progress)

    q1.observe(check_q1, names="value")

    display(question_label, q1, output1)


def run_few_shot_full_dataset(rows, progress):
    """
    Run few-shot classification on full dataset and display results.

    Args:
        rows: Dataset as list of (text, label) tuples
        progress: Progress dictionary to save metrics

    Returns:
        metrics_fs: Dictionary with performance metrics
    """
    print("=" * 70)
    print(f"🚀 RUNNING FEW-SHOT ON ALL {len(rows)} EXAMPLES")
    print("=" * 70)
    print("\nThis will classify all movie reviews using few-shot prompting...")
    print("💡 Using 3 demonstration examples for each classification")
    print("⏳ Please wait...\n")

    # Define demonstrations (same for all examples)
    demonstrations = [
        ("I loved the film, it was wonderful!", "positive"),
        ("The acting was bad and the script was messy.", "negative"),
        ("A charming and uplifting story.", "positive"),
    ]

    # Run few-shot on entire dataset
    pred_fs, metas_fs = run_few_shot(rows, demos=demonstrations)
    true = [y for _, y in rows]
    metrics_fs = precision_recall_f1(true, pred_fs)
    metrics_fs.update({"calls": len(pred_fs)})

    # Save to progress
    progress.setdefault("metrics", {})["few_shot"] = metrics_fs
    save_progress(progress)

    print("✅ Few-shot classification complete!\n")

    # Display metrics in a table
    print("📊 RESULTS SUMMARY:\n")

    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Score": [
            f"{metrics_fs['accuracy']:.1%}",
            f"{metrics_fs['precision']:.1%}",
            f"{metrics_fs['recall']:.1%}",
            f"{metrics_fs['f1']:.1%}",
        ],
        "Description": [
            "Overall correct predictions",
            "Positive precision",
            "Positive recall",
            "Harmonic mean",
        ],
    }

    df_metrics = pd.DataFrame(metrics_data)
    display(
        df_metrics.style.set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#8b5cf6"),  # Purple for few-shot
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                        ("padding", "10px"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "left"),
                        ("padding", "8px"),
                        ("border-bottom", "1px solid #ddd"),
                    ],
                },
            ]
        ).hide(axis="index")
    )

    # Visualize metrics
    print("\n" + "=" * 70)
    print("📈 VISUALIZATION")
    print("=" * 70 + "\n")

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    metrics_values = [
        metrics_fs["accuracy"],
        metrics_fs["precision"],
        metrics_fs["recall"],
        metrics_fs["f1"],
    ]

    bars = ax.bar(
        metrics_names,
        metrics_values,
        color="#8b5cf6",
        alpha=0.8,
        edgecolor="#6d28d9",
        linewidth=2,
    )
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title(
        "Few-shot Performance on Full Dataset", fontsize=15, fontweight="bold", pad=20
    )

    # Target line
    ax.axhline(
        y=0.6,
        color="#10b981",
        linestyle="--",
        linewidth=2.5,
        label="Target: 60%",
        alpha=0.8,
    )
    ax.legend(fontsize=11, loc="lower right")

    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=1)
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.03,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
    plt.tight_layout()
    plt.show()

    # Detailed breakdown
    print("\n" + "=" * 70)
    print("📋 DETAILED BREAKDOWN")
    print("=" * 70)
    print(f"\n   ✅ True Positives: {metrics_fs['tp']}")
    print(f"   ✅ True Negatives: {metrics_fs['tn']}")
    print(f"   ❌ False Positives: {metrics_fs['fp']}")
    print(f"   ❌ False Negatives: {metrics_fs['fn']}")
    print(f"   📞 Total API calls: {metrics_fs['calls']}\n")

    # Check if target reached
    print("=" * 70)
    if metrics_fs["accuracy"] >= 0.6:
        print("🎉 SUCCESS! Few-shot reached the target accuracy of 60%!")
    else:
        print("⚠️  Few-shot accuracy is below 60%.")
        print("💡 Tip: Chain-of-Thought prompting might improve performance.")
    print("=" * 70)

    print("\n💡 Continue to the next section to explore Chain-of-Thought prompting!")
    print(
        "   After completing all methods, check the Comparison section for full analysis.\n"
    )

    return metrics_fs


# ============================================================================
# SECTION 3: CHAIN-OF-THOUGHT (CoT) HELPERS
# ============================================================================


def display_cot_example(rows):
    """
    Display a complete Chain-of-Thought classification example with styling.

    Args:
        rows: Dataset as list of (text, label) tuples
    """
    print("🎬 Let's see how Chain-of-Thought prompting works!\n")

    # Step 1: Choose an example
    sentence = "The movie was absolutely wonderful and full of emotion."
    expected = "positive"

    print(f"📄 Input sentence:")
    print(f'   "{sentence}"\n')
    print(f"🎯 Expected sentiment: {expected.upper()}\n")

    # Step 2: Build CoT prompt
    prompt = build_cot_prompt(sentence)

    print("=" * 70)
    print("🔍 PROMPT SENT TO MODEL:")
    print("=" * 70)
    print(prompt)
    print("=" * 70)

    # Step 3: Call the model
    print("\n⏳ Calling model...\n")
    model_output, meta = call_llm(prompt, temperature=0.0, max_tokens=300)

    print("=" * 70)
    print("🤖 MODEL RESPONSE (Full Reasoning):")
    print("=" * 70)
    print(model_output)
    print("=" * 70)

    # Step 4: Extract final answer
    if model_output and model_output.strip():
        last_word = model_output.strip().split()[-1]
        prediction = normalize_label(last_word)
    else:
        prediction = "negative"

    print(f"\n📊 Extracted Answer:")
    print(f"   Last word: {last_word}")
    print(f"   Normalized: {prediction.upper()}")
    print(f"   Latency: {meta.get('latency_s', 0):.3f}s\n")

    # Step 5: Display results in table
    data = {
        "Sentence": [sentence],
        "Model Prediction": [prediction.capitalize()],
        "Expected": [expected.capitalize()],
        "Result": [
            "✅ Correct" if prediction.lower() == expected.lower() else "❌ Incorrect"
        ],
    }

    df = pd.DataFrame(data)
    styled_df = df.style.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#ec4899"),  # Pink for CoT
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("padding", "10px"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "left"),
                    ("padding", "8px"),
                    ("border-bottom", "1px solid #ddd"),
                ],
            },
        ]
    ).hide(axis="index")

    display(styled_df)

    # Step 6: Explain what happened
    print()
    if prediction.lower() == expected.lower():
        print("✅ Success! The model reasoned step-by-step before answering.")
        print(
            "   💡 Notice how the reasoning process helps arrive at the correct answer."
        )
    else:
        print("❌ The model made an error despite showing reasoning.")
        print(
            "   💡 Tip: Self-consistency might help by running multiple reasoning paths!"
        )


def create_cot_interactive():
    """
    Interactive test - uses standard input() for consistent styling.
    """
    print("🎯 Test Chain-of-Thought prompting with your own sentence!\n")

    sentence = input("Enter a sentence to classify: ")

    if sentence.strip():
        # Build CoT prompt
        prompt = build_cot_prompt(sentence)

        print(f"\n{'=' * 70}")
        print("🔍 PROMPT SENT TO MODEL:")
        print(f"{'=' * 70}")
        print(prompt)
        print(f"{'=' * 70}")

        # Call model
        print("\n⏳ Calling model...\n")
        model_output, meta = call_llm(prompt, temperature=0.0, max_tokens=300)

        print("=" * 70)
        print("🤖 MODEL REASONING:")
        print("=" * 70)
        print(model_output)
        print("=" * 70)

        # Extract answer
        if model_output and model_output.strip():
            last_word = model_output.strip().split()[-1]
            prediction = normalize_label(last_word)
        else:
            prediction = "negative"

        print(f"\n📊 Final Answer: {prediction.upper()}")
        print(f"⏱️  Latency: {meta.get('latency_s', 0):.3f}s")

        print(
            "\n💡 Notice how the model explains its reasoning before giving the answer!"
        )
    else:
        print("❌ No input provided")


def create_cot_quiz(progress):
    """
    Create and display the CoT quiz widget.

    Args:
        progress: Progress dictionary to save quiz results
    """
    # Display quiz header
    display(
        HTML("""
    <div style="background:#dbeafe; border-left:5px solid #2563eb; padding:14px 18px; margin:16px 0; border-radius:6px; color:#1e3a8a;">
        <strong style="font-size:16px;">🔍 Knowledge Check</strong><br>
        <span style="font-size:14px;">Test your understanding of Chain-of-Thought prompting before moving forward.</span>
    </div>
    """)
    )

    # Question label
    question_label = widgets.HTML(
        value="<b style='font-size:15px; color:#1f2937;'>❓ What is the main advantage of Chain-of-Thought prompting?</b>",
        layout=widgets.Layout(margin="10px 0 8px 0"),
    )

    # Radio buttons
    q1 = widgets.RadioButtons(
        options=[
            "Provide multiple examples to guide the model",
            "Run the model multiple times and pick the best answer",
            "Ask the model to explain its reasoning step-by-step",
        ],
        value=None,
        description="",
        style={"description_width": "0px"},
        layout=widgets.Layout(width="100%", margin="10px 0"),
    )

    # Apply custom CSS
    display(
        HTML("""
    <style>
    .widget-radio-box label {
        color: #1f2937 !important;
        font-size: 14px !important;
        line-height: 1.8 !important;
    }
    .widget-radio-box input[type="radio"] {
        margin-right: 10px !important;
    }
    </style>
    """)
    )

    output1 = widgets.Output()

    def check_q1(change):
        if change["new"] is None:
            return

        with output1:
            clear_output()
            is_correct, feedback_html = check_answer("cot_q1", change["new"])
            display(HTML(feedback_html))

            if is_correct:
                quiz = progress.setdefault("quiz", {})
                quiz["quiz_cot_q1"] = "correct"
                save_progress(progress)

    q1.observe(check_q1, names="value")

    display(question_label, q1, output1)


def run_cot_full_dataset(rows, progress):
    """
    Run Chain-of-Thought classification on full dataset and display results.

    Args:
        rows: Dataset as list of (text, label) tuples
        progress: Progress dictionary to save metrics

    Returns:
        metrics_cot: Dictionary with performance metrics
    """
    print("=" * 70)
    print(f"🚀 RUNNING CHAIN-OF-THOUGHT ON ALL {len(rows)} EXAMPLES")
    print("=" * 70)
    print("\nThis will classify all movie reviews using Chain-of-Thought prompting...")
    print("💡 Each classification includes step-by-step reasoning")
    print("⏳ Please wait...\n")

    # Run CoT on entire dataset
    pred_cot, metas_cot = run_cot(rows)
    true = [y for _, y in rows]
    metrics_cot = precision_recall_f1(true, pred_cot)
    metrics_cot.update({"calls": len(pred_cot)})

    # Save to progress
    progress.setdefault("metrics", {})["cot"] = metrics_cot
    save_progress(progress)

    print("✅ Chain-of-Thought classification complete!\n")

    # Display metrics in a table
    print("📊 RESULTS SUMMARY:\n")

    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Score": [
            f"{metrics_cot['accuracy']:.1%}",
            f"{metrics_cot['precision']:.1%}",
            f"{metrics_cot['recall']:.1%}",
            f"{metrics_cot['f1']:.1%}",
        ],
        "Description": [
            "Overall correct predictions",
            "Positive precision",
            "Positive recall",
            "Harmonic mean",
        ],
    }

    df_metrics = pd.DataFrame(metrics_data)
    display(
        df_metrics.style.set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#ec4899"),  # Pink for CoT
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                        ("padding", "10px"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "left"),
                        ("padding", "8px"),
                        ("border-bottom", "1px solid #ddd"),
                    ],
                },
            ]
        ).hide(axis="index")
    )

    # Visualize metrics
    print("\n" + "=" * 70)
    print("📈 VISUALIZATION")
    print("=" * 70 + "\n")

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    metrics_values = [
        metrics_cot["accuracy"],
        metrics_cot["precision"],
        metrics_cot["recall"],
        metrics_cot["f1"],
    ]

    bars = ax.bar(
        metrics_names,
        metrics_values,
        color="#ec4899",
        alpha=0.8,
        edgecolor="#be185d",
        linewidth=2,
    )
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title(
        "Chain-of-Thought Performance on Full Dataset",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    # Target line
    ax.axhline(
        y=0.6,
        color="#10b981",
        linestyle="--",
        linewidth=2.5,
        label="Target: 60%",
        alpha=0.8,
    )
    ax.legend(fontsize=11, loc="lower right")

    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=1)
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.03,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
    plt.tight_layout()
    plt.show()

    # Detailed breakdown
    print("\n" + "=" * 70)
    print("📋 DETAILED BREAKDOWN")
    print("=" * 70)
    print(f"\n   ✅ True Positives: {metrics_cot['tp']}")
    print(f"   ✅ True Negatives: {metrics_cot['tn']}")
    print(f"   ❌ False Positives: {metrics_cot['fp']}")
    print(f"   ❌ False Negatives: {metrics_cot['fn']}")
    print(f"   📞 Total API calls: {metrics_cot['calls']}\n")

    # Check if target reached
    print("=" * 70)
    if metrics_cot["accuracy"] >= 0.6:
        print("🎉 SUCCESS! Chain-of-Thought reached the target accuracy of 60%!")
    else:
        print("⚠️  Chain-of-Thought accuracy is below 60%.")
        print("💡 Tip: Self-consistency might improve performance further.")
    print("=" * 70)

    print("\n💡 Continue to the next section to explore Self-Consistency!")
    print(
        "   After completing all methods, check the Comparison section for full analysis.\n"
    )

    return metrics_cot


# ============================================================================
# SECTION 4: SELF-CONSISTENCY HELPERS
# ============================================================================


def display_self_consistency_example(rows, k=5):
    """
    Display a complete Self-Consistency classification example with styling.

    Args:
        rows: Dataset as list of (text, label) tuples
        k: Number of reasoning paths to sample
    """
    print("🎬 Let's see how Self-Consistency works!\n")

    # Step 1: Choose an example
    sentence = "The movie was absolutely wonderful and full of emotion."
    expected = "positive"

    print(f"📄 Input sentence:")
    print(f'   "{sentence}"\n')
    print(f"🎯 Expected sentiment: {expected.upper()}\n")

    # Step 2: Run multiple reasoning paths
    print(f"🔄 Running {k} reasoning paths with temperature=0.7...\n")

    votes = []
    all_outputs = []

    for i in range(k):
        print(f"{'=' * 70}")
        print(f"🧠 REASONING PATH {i + 1}/{k}")
        print(f"{'=' * 70}")

        prompt = build_cot_prompt(sentence)
        model_output, meta = call_llm(prompt, temperature=0.7, max_tokens=300)

        print(model_output)
        print(f"\n⏱️  Latency: {meta.get('latency_s', 0):.3f}s")

        # Extract answer
        if model_output and model_output.strip():
            last_word = model_output.strip().split()[-1]
            prediction = normalize_label(last_word)
        else:
            prediction = "negative"

        votes.append(prediction)
        all_outputs.append(model_output)
        print(f"📊 Extracted answer: {prediction.upper()}\n")

    # Step 3: Majority voting
    print("=" * 70)
    print("🗳️  MAJORITY VOTING")
    print("=" * 70)

    from collections import Counter

    vote_counts = Counter(votes)

    print("\nVote distribution:")
    for label, count in vote_counts.most_common():
        percentage = (count / k) * 100
        bar = "█" * count + "░" * (k - count)
        print(f"   {label.capitalize()}: {count}/{k} ({percentage:.0f}%) {bar}")

    final_answer = vote_counts.most_common(1)[0][0]
    print(f"\n🎯 Final Answer (Majority): {final_answer.upper()}")

    # Step 4: Display results in table
    print("\n📊 Results Summary:\n")

    data = {
        "Sentence": [sentence],
        "Model Prediction": [final_answer.capitalize()],
        "Expected": [expected.capitalize()],
        "Result": [
            "✅ Correct" if final_answer.lower() == expected.lower() else "❌ Incorrect"
        ],
        "Vote Split": [f"{vote_counts[final_answer]}/{k}"],
    }

    df = pd.DataFrame(data)
    styled_df = df.style.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#f59e0b"),  # Orange for Self-Consistency
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("padding", "10px"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "left"),
                    ("padding", "8px"),
                    ("border-bottom", "1px solid #ddd"),
                ],
            },
        ]
    ).hide(axis="index")

    display(styled_df)

    # Step 5: Explain what happened
    print()
    if final_answer.lower() == expected.lower():
        print("✅ Success! Multiple reasoning paths reached consensus.")
        print(
            f"   💡 {vote_counts[final_answer]} out of {k} paths agreed on the answer."
        )
    else:
        print("❌ The majority vote was incorrect.")
        print("   💡 This can happen with challenging examples or low k values.")

    print(f"\n💰 Cost: {k} API calls (vs. 1 for zero-shot/few-shot/CoT)")


def create_self_consistency_interactive(k=5):
    """
    Interactive test - uses standard input() for consistent styling.
    """
    print("🎯 Test Self-Consistency with your own sentence!\n")

    sentence = input("Enter a sentence to classify: ")

    if sentence.strip():
        print(f"\n🔄 Running {k} reasoning paths...\n")

        votes = []

        for i in range(k):
            print(f"{'=' * 70}")
            print(f"🧠 PATH {i + 1}/{k}")
            print(f"{'=' * 70}")

            prompt = build_cot_prompt(sentence)
            model_output, meta = call_llm(prompt, temperature=0.7, max_tokens=300)

            print(model_output)

            # Extract answer
            if model_output and model_output.strip():
                last_word = model_output.strip().split()[-1]
                prediction = normalize_label(last_word)
            else:
                prediction = "negative"

            votes.append(prediction)
            print(f"\n📊 Answer: {prediction.upper()}")
            print(f"⏱️  Latency: {meta.get('latency_s', 0):.3f}s\n")

        # Majority voting
        print("=" * 70)
        print("🗳️  MAJORITY VOTING")
        print("=" * 70)

        from collections import Counter

        vote_counts = Counter(votes)

        print("\nVote distribution:")
        for label, count in vote_counts.most_common():
            percentage = (count / k) * 100
            bar = "█" * count + "░" * (k - count)
            print(f"   {label.capitalize()}: {count}/{k} ({percentage:.0f}%) {bar}")

        final_answer = vote_counts.most_common(1)[0][0]
        print(f"\n🎯 Final Answer: {final_answer.upper()}")
        print(f"💰 Total cost: {k} API calls")

        print(
            "\n💡 Notice how multiple reasoning paths can increase confidence in the answer!"
        )
    else:
        print("❌ No input provided")


def create_self_consistency_quiz(progress):
    """
    Create and display the Self-Consistency quiz widget.

    Args:
        progress: Progress dictionary to save quiz results
    """
    # Display quiz header
    display(
        HTML("""
    <div style="background:#dbeafe; border-left:5px solid #2563eb; padding:14px 18px; margin:16px 0; border-radius:6px; color:#1e3a8a;">
        <strong style="font-size:16px;">🔍 Knowledge Check</strong><br>
        <span style="font-size:14px;">Test your understanding of Self-Consistency before moving forward.</span>
    </div>
    """)
    )

    # Question label
    question_label = widgets.HTML(
        value="<b style='font-size:15px; color:#1f2937;'>❓ What is the key mechanism of Self-Consistency?</b>",
        layout=widgets.Layout(margin="10px 0 8px 0"),
    )

    # Radio buttons
    q1 = widgets.RadioButtons(
        options=[
            "Run multiple reasoning paths and use majority vote",
            "Provide multiple examples in a single prompt",
            "Use a more powerful model for better accuracy",
        ],
        value=None,
        description="",
        style={"description_width": "0px"},
        layout=widgets.Layout(width="100%", margin="10px 0"),
    )

    # Apply custom CSS
    display(
        HTML("""
    <style>
    .widget-radio-box label {
        color: #1f2937 !important;
        font-size: 14px !important;
        line-height: 1.8 !important;
    }
    .widget-radio-box input[type="radio"] {
        margin-right: 10px !important;
    }
    </style>
    """)
    )

    output1 = widgets.Output()

    def check_q1(change):
        if change["new"] is None:
            return

        with output1:
            clear_output()
            is_correct, feedback_html = check_answer(
                "self_consistency_q1", change["new"]
            )
            display(HTML(feedback_html))

            if is_correct:
                quiz = progress.setdefault("quiz", {})
                quiz["quiz_self_consistency_q1"] = "correct"
                save_progress(progress)

    q1.observe(check_q1, names="value")

    display(question_label, q1, output1)


def run_self_consistency_full_dataset(rows, progress, k=5):
    """
    Run Self-Consistency classification on full dataset and display results.

    Args:
        rows: Dataset as list of (text, label) tuples
        progress: Progress dictionary to save metrics
        k: Number of reasoning paths per example

    Returns:
        metrics_sc: Dictionary with performance metrics
    """
    print("=" * 70)
    print(f"🚀 RUNNING SELF-CONSISTENCY ON ALL {len(rows)} EXAMPLES")
    print("=" * 70)
    print("\nThis will classify all movie reviews using Self-Consistency...")
    print(f"💡 Each classification uses {k} reasoning paths + majority voting")
    print("⚠️  This will take longer due to multiple samples per example")
    print("⏳ Please wait...\n")

    # Run Self-Consistency on entire dataset
    pred_sc, metas_sc = run_self_consistency(rows, k=k)
    true = [y for _, y in rows]
    metrics_sc = precision_recall_f1(true, pred_sc)
    metrics_sc.update({"calls": len(metas_sc)})  # Note: this is k * len(dataset)

    # Save to progress
    progress.setdefault("metrics", {})["self_consistency"] = metrics_sc
    save_progress(progress)

    print("✅ Self-Consistency classification complete!\n")

    # Display metrics in a table
    print("📊 RESULTS SUMMARY:\n")

    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Score": [
            f"{metrics_sc['accuracy']:.1%}",
            f"{metrics_sc['precision']:.1%}",
            f"{metrics_sc['recall']:.1%}",
            f"{metrics_sc['f1']:.1%}",
        ],
        "Description": [
            "Overall correct predictions",
            "Positive precision",
            "Positive recall",
            "Harmonic mean",
        ],
    }

    df_metrics = pd.DataFrame(metrics_data)
    display(
        df_metrics.style.set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#f59e0b"),  # Orange for Self-Consistency
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                        ("padding", "10px"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "left"),
                        ("padding", "8px"),
                        ("border-bottom", "1px solid #ddd"),
                    ],
                },
            ]
        ).hide(axis="index")
    )

    # Visualize metrics
    print("\n" + "=" * 70)
    print("📈 VISUALIZATION")
    print("=" * 70 + "\n")

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    metrics_values = [
        metrics_sc["accuracy"],
        metrics_sc["precision"],
        metrics_sc["recall"],
        metrics_sc["f1"],
    ]

    bars = ax.bar(
        metrics_names,
        metrics_values,
        color="#f59e0b",
        alpha=0.8,
        edgecolor="#d97706",
        linewidth=2,
    )
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title(
        f"Self-Consistency Performance on Full Dataset (k={k})",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    # Target line
    ax.axhline(
        y=0.6,
        color="#10b981",
        linestyle="--",
        linewidth=2.5,
        label="Target: 60%",
        alpha=0.8,
    )
    ax.legend(fontsize=11, loc="lower right")

    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=1)
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.03,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
    plt.tight_layout()
    plt.show()

    # Detailed breakdown
    print("\n" + "=" * 70)
    print("📋 DETAILED BREAKDOWN")
    print("=" * 70)
    print(f"\n   ✅ True Positives: {metrics_sc['tp']}")
    print(f"   ✅ True Negatives: {metrics_sc['tn']}")
    print(f"   ❌ False Positives: {metrics_sc['fp']}")
    print(f"   ❌ False Negatives: {metrics_sc['fn']}")
    print(
        f"   📞 Total API calls: {metrics_sc['calls']} ({len(rows)} examples × {k} paths)"
    )
    print(f"   💰 Cost multiplier: {k}× compared to single-path methods\n")

    # Check if target reached
    print("=" * 70)
    if metrics_sc["accuracy"] >= 0.6:
        print("🎉 SUCCESS! Self-Consistency reached the target accuracy of 60%!")
    else:
        print("⚠️  Self-Consistency accuracy is below 60%.")
    print("=" * 70)

    print("\n🎯 You've completed all four prompting methods!")
    print("   Continue to the Comparison section to analyze all results together.\n")

    return metrics_sc


# ============================================================================
# SECTION 5: COMPARISON & REFLECTION HELPERS
# ============================================================================


def display_method_comparison(progress):
    """
    Display complete comparison of all prompting methods with tables, charts, and reflections.

    Args:
        progress: Progress dictionary containing all metrics and quiz results
    """
    print("=" * 80)
    print("🎯 COMPARISON OF PROMPTING METHODS")
    print("=" * 80)
    print()

    # ========================================
    # 1. CONCEPTUAL COMPARISON TABLE
    # ========================================

    comparison_data = {
        "Aspect": [
            "Examples provided",
            "Prompt length",
            "Reasoning shown",
            "Multiple samples",
            "Token cost",
            "Typical use case",
        ],
        "Zero-shot": [
            "❌ None",
            "🟢 Short",
            "❌ No",
            "❌ No",
            "💰 Lowest",
            "Clear, simple tasks",
        ],
        "Few-shot": [
            "✅ 2-10 examples",
            "🟡 Medium",
            "❌ No",
            "❌ No",
            "💰💰 Medium",
            "Pattern learning",
        ],
        "Chain-of-Thought": [
            "❌ None",
            "🟡 Medium",
            "✅ Yes",
            "❌ No",
            "💰💰 Medium",
            "Complex reasoning",
        ],
        "Self-Consistency": [
            "❌ None",
            "🔴 Long",
            "✅ Yes",
            "✅ Yes (k=5)",
            "💰💰💰 Highest",
            "High reliability",
        ],
    }

    df_comparison = pd.DataFrame(comparison_data)

    styled_comparison = df_comparison.style.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#3b82f6"),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("padding", "12px"),
                    ("font-size", "13px"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "left"),
                    ("padding", "10px"),
                    ("border-bottom", "1px solid #e5e7eb"),
                    ("font-size", "13px"),
                ],
            },
            {
                "selector": "td:first-child",
                "props": [("font-weight", "bold"), ("background-color", "#f9fafb")],
            },
        ]
    ).hide(axis="index")

    display(styled_comparison)

    # ========================================
    # 2. PERFORMANCE COMPARISON TABLE
    # ========================================

    print("\n" + "=" * 80)
    print("📈 PERFORMANCE COMPARISON")
    print("=" * 80)
    print()

    metrics = progress.get("metrics", {})

    methods = ["zero_shot", "few_shot", "cot", "self_consistency"]
    method_names = ["Zero-shot", "Few-shot", "Chain-of-Thought", "Self-Consistency"]

    performance_data = {
        "Method": method_names,
        "Accuracy": [f"{metrics.get(m, {}).get('accuracy', 0):.1%}" for m in methods],
        "Precision": [f"{metrics.get(m, {}).get('precision', 0):.1%}" for m in methods],
        "Recall": [f"{metrics.get(m, {}).get('recall', 0):.1%}" for m in methods],
        "F1-score": [f"{metrics.get(m, {}).get('f1', 0):.1%}" for m in methods],
        "API Calls": [metrics.get(m, {}).get("calls", 0) for m in methods],
    }

    df_performance = pd.DataFrame(performance_data)

    # Highlight best performance
    def highlight_max(s):
        """Highlight max value in each column (except Method and API Calls)"""
        if s.name in ["Accuracy", "Precision", "Recall", "F1-score"]:
            values = [float(v.strip("%")) for v in s]
            is_max = [v == max(values) for v in values]
            return [
                "background-color: #d1fae5; font-weight: bold" if m else ""
                for m in is_max
            ]
        return ["" for _ in s]

    styled_performance = (
        df_performance.style.apply(highlight_max)
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#3b82f6"),
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                        ("padding", "12px"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "center"),
                        ("padding", "10px"),
                        ("border-bottom", "1px solid #e5e7eb"),
                    ],
                },
                {
                    "selector": "td:first-child",
                    "props": [("text-align", "left"), ("font-weight", "bold")],
                },
            ]
        )
        .hide(axis="index")
    )

    display(styled_performance)

    # ========================================
    # 3. VISUALIZATIONS
    # ========================================

    print("\n📊 Visualizing method comparison...\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Chart 1: Accuracy comparison
    accuracies = [metrics.get(m, {}).get("accuracy", 0) for m in methods]
    colors = ["#3b82f6", "#8b5cf6", "#ec4899", "#f59e0b"]

    bars = ax1.bar(
        method_names,
        accuracies,
        color=colors,
        alpha=0.8,
        edgecolor="#1e40af",
        linewidth=2,
    )
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Accuracy", fontsize=13, fontweight="bold")
    ax1.set_title("Accuracy Comparison", fontsize=15, fontweight="bold", pad=15)
    ax1.axhline(
        y=0.6,
        color="#10b981",
        linestyle="--",
        linewidth=2,
        label="Target: 60%",
        alpha=0.7,
    )
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3, linestyle=":", linewidth=1)
    ax1.set_axisbelow(True)

    # Add value labels
    for bar, val in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    # Chart 2: Cost comparison (API calls)
    api_calls = [metrics.get(m, {}).get("calls", 0) for m in methods]

    bars2 = ax2.bar(
        method_names,
        api_calls,
        color=colors,
        alpha=0.8,
        edgecolor="#1e40af",
        linewidth=2,
    )
    ax2.set_ylabel("API Calls", fontsize=13, fontweight="bold")
    ax2.set_title(
        "Cost Comparison (Total API Calls)", fontsize=15, fontweight="bold", pad=15
    )
    ax2.grid(axis="y", alpha=0.3, linestyle=":", linewidth=1)
    ax2.set_axisbelow(True)

    # Add value labels
    for bar, val in zip(bars2, api_calls):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{val}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    plt.tight_layout()
    plt.show()

    # ========================================
    # 4. KEY INSIGHTS
    # ========================================

    print("\n" + "=" * 80)
    print("💡 KEY INSIGHTS")
    print("=" * 80)

    best_accuracy_idx = accuracies.index(max(accuracies))
    most_expensive_idx = api_calls.index(max(api_calls))

    print(
        f"\n🏆 Best Accuracy: {method_names[best_accuracy_idx]} ({accuracies[best_accuracy_idx]:.1%})"
    )
    print(
        f"💰 Most Expensive: {method_names[most_expensive_idx]} ({api_calls[most_expensive_idx]} API calls)"
    )
    print(f"⚡ Most Efficient: {method_names[0]} ({api_calls[0]} API calls)")
    print("=" * 80)

    # ========================================
    # 5. INTERACTIVE REFLECTION SECTION
    # ========================================

    display(
        HTML("""
    <div style="background:#fff9db; border-left:5px solid #f59e0b; padding:16px 20px; margin:16px 0; border-radius:6px; color:#78350f; line-height:1.6;">
        <strong style="font-size:15px;">📝 YOUR REFLECTIONS</strong><br>
        <span style="font-size:14px;">Based on the results above, answer the following reflection questions. Your answers will be saved to your progress file.</span>
    </div>
    """)
    )

    # Create text areas for reflections
    reflection_q1 = widgets.Textarea(
        value=progress.get("quiz", {}).get("reflection_best_method", ""),
        placeholder="Which method performed best on this dataset and why do you think that is?",
        description="",
        layout=widgets.Layout(width="100%", height="100px"),
        style={"description_width": "0px"},
    )

    reflection_q2 = widgets.Textarea(
        value=progress.get("quiz", {}).get("reflection_trade_offs", ""),
        placeholder="What are the key trade-offs between accuracy and cost? When would you choose a more expensive method?",
        description="",
        layout=widgets.Layout(width="100%", height="100px"),
        style={"description_width": "0px"},
    )

    reflection_q3 = widgets.Textarea(
        value=progress.get("quiz", {}).get("reflection_real_world", ""),
        placeholder="If you were building a real-world sentiment analysis system, which method would you choose and why?",
        description="",
        layout=widgets.Layout(width="100%", height="100px"),
        style={"description_width": "0px"},
    )

    # Labels for each question
    q1_label = widgets.HTML(
        value="<b style='color:#1f2937; font-size:14px;'>1️⃣ Which method performed best on this dataset and why?</b>",
        layout=widgets.Layout(margin="16px 0 8px 0"),
    )

    q2_label = widgets.HTML(
        value="<b style='color:#1f2937; font-size:14px;'>2️⃣ What are the key trade-offs between accuracy and cost?</b>",
        layout=widgets.Layout(margin="16px 0 8px 0"),
    )

    q3_label = widgets.HTML(
        value="<b style='color:#1f2937; font-size:14px;'>3️⃣ For a real-world system, which method would you choose?</b>",
        layout=widgets.Layout(margin="16px 0 8px 0"),
    )

    # Save button
    save_button = widgets.Button(
        description="💾 Save Reflections",
        button_style="success",
        layout=widgets.Layout(width="200px", height="40px", margin="20px 0"),
    )

    output_feedback = widgets.Output()

    def save_reflections(b):
        """Save reflection answers to progress file"""
        with output_feedback:
            clear_output()

            # CRITICAL: Load fresh progress from disk
            prog = load_progress()
            quiz_data = prog.setdefault("quiz", {})

            # Save to fresh quiz data
            quiz_data["reflection_best_method"] = reflection_q1.value.strip()
            quiz_data["reflection_trade_offs"] = reflection_q2.value.strip()
            quiz_data["reflection_real_world"] = reflection_q3.value.strip()

            # Check if all are filled
            all_filled = all(
                [
                    reflection_q1.value.strip(),
                    reflection_q2.value.strip(),
                    reflection_q3.value.strip(),
                ]
            )

            if all_filled:
                quiz_data["quiz_conclusion"] = "complete"
                save_progress(prog)
                display(
                    HTML("""
                    <div style="background:#d1fae5; border-left:4px solid #10b981; padding:12px 16px; border-radius:6px; margin-top:12px; color:#065f46;">
                        <strong>✅ Reflections saved successfully!</strong><br>
                        Your answers have been recorded. You can now run <code>python src/verify.py</code> to validate your completion.
                    </div>
                """)
                )
            else:
                save_progress(prog)
                display(
                    HTML("""
                    <div style="background:#fef3c7; border-left:4px solid #f59e0b; padding:12px 16px; border-radius:6px; margin-top:12px; color:#78350f;">
                        <strong>⚠️ Partial save</strong><br>
                        Some reflections are empty. Please complete all questions for full credit.
                    </div>
                """)
                )

    save_button.on_click(save_reflections)

    # Display all reflection widgets
    display(q1_label, reflection_q1)
    display(q2_label, reflection_q2)
    display(q3_label, reflection_q3)
    display(save_button, output_feedback)

    print("\n" + "=" * 80)

    # ============================================================================


# SECTION 6: PROGRESS TRACKER & VALIDATION HELPER
# ============================================================================


def run_progress_tracker_and_validation(progress, project_root):
    """
    Run automated validation and display detailed progress breakdown.

    Args:
        progress: Progress dictionary
        project_root: Path to project root directory
    """
    import subprocess
    import sys

    print("=" * 80)
    print("📊 RUNNING AUTOMATED VALIDATION")
    print("=" * 80)
    print("\nChecking your progress against all requirements...\n")

    # Run verify.py with UTF-8 encoding
    try:
        r = subprocess.run(
            [sys.executable, "src/verify.py"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(project_root),
        )
        print(r.stdout)
        if r.stderr:
            print("⚠️  Errors:", r.stderr)
    except Exception as e:
        print(f"❌ Could not run verify.py automatically: {e}")
        print("💡 Run manually: python src/verify.py")

    # Show detailed progress
    print("\n" + "=" * 80)
    print("📋 DETAILED PROGRESS BREAKDOWN")
    print("=" * 80)

    prog = load_progress()
    metrics = prog.get("metrics", {})
    quiz = prog.get("quiz", {})

    # Required methods
    required_methods = ["zero_shot", "few_shot", "cot", "self_consistency"]

    print("\n🔬 METHODS COMPLETED:")
    for method in required_methods:
        if method in metrics:
            m = metrics[method]
            status = "✅"
            print(
                f"   {status} {method.replace('_', '-').title()}: "
                f"Accuracy={m.get('accuracy', 0):.1%}, "
                f"F1={m.get('f1', 0):.1%}, "
                f"Calls={m.get('calls', 0)}"
            )
        else:
            print(f"   ❌ {method.replace('_', '-').title()}: Not completed")

    # Required quiz questions
    required_quizzes = [
        "quiz_zero_shot_q1",
        "quiz_few_shot_q1",
        "quiz_cot_q1",
        "quiz_self_consistency_q1",
    ]

    print("\n🧠 QUIZ QUESTIONS:")
    for q in required_quizzes:
        status = "✅" if quiz.get(q) == "correct" else "❌"
        method = q.replace("quiz_", "").replace("_q1", "").replace("_", "-").title()
        print(f"   {status} {method}")

    # Required reflections
    required_reflections = [
        "reflection_best_method",
        "reflection_trade_offs",
        "reflection_real_world",
    ]

    print("\n💭 REFLECTION QUESTIONS:")
    for r in required_reflections:
        status = "✅" if quiz.get(r) and quiz.get(r).strip() else "❌"
        name = r.replace("reflection_", "").replace("_", " ").title()
        print(f"   {status} {name}")

    # Summary
    print("\n" + "=" * 80)
    total_checks = (
        len(required_methods) + len(required_quizzes) + len(required_reflections)
    )
    completed = (
        sum(1 for m in required_methods if m in metrics)
        + sum(1 for q in required_quizzes if quiz.get(q) == "correct")
        + sum(1 for r in required_reflections if quiz.get(r) and quiz.get(r).strip())
    )

    print(f"📈 COMPLETION: {completed}/{total_checks} ({completed / total_checks:.0%})")
    print("=" * 80)

    # Check for receipt
    receipt_path = project_root / "progress" / "receipt.json"
    if receipt_path.exists():
        import json

        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") == "PASS":
            print("\n🎉 CONGRATULATIONS! You have successfully completed the lesson!")
            print(f"📄 Receipt generated: {receipt_path}")
            print(f"⏰ Timestamp: {receipt.get('timestamp')}")
        else:
            print("\n⚠️  Validation failed. See output above for details.")
    else:
        print("\n⚠️  No receipt generated yet. Complete all requirements to pass.")

    print("\n💡 To re-validate, run: python src/verify.py")
    print("=" * 80)
