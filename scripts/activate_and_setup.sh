#!/usr/bin/env bash

# Run the setup script
source ./scripts/setup_venv.sh

# Activate the virtual environment
echo ""
echo "🔌 Activating virtual environment..."
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
    echo "✅ Virtual environment activated! (you should see (.venv) in your prompt)"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated! (you should see (.venv) in your prompt)"
else
    echo "⚠️  Could not find activation script."
    echo "Please activate manually with:"
    echo "   source .venv/Scripts/activate  (Windows Git Bash)"
    echo "   source .venv/bin/activate       (Mac/Linux)"
fi

echo ""
echo "📌 Next steps:"
echo "   1. ✅ Environment is ready and activated"
echo "   2. Install Ollama and download model (see Start Guide - index.html)"
echo "   3. Start Jupyter: jupyter notebook notebooks/lesson_all_in_one.ipynb"
echo ""