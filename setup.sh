#!/usr/bin/env bash
# Bootstrap script - makes other scripts executable and runs setup

echo "🔧 Making scripts executable..."
chmod +x scripts/*.sh

echo "🚀 Running environment setup..."
bash ./scripts/setup_venv.sh

# Now activate in current shell
echo ""
echo "🔌 Activating virtual environment..."
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
    echo "✅ Virtual environment activated!"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated!"
fi

echo ""
echo "📌 Next steps:"
echo "   1. ✅ Environment is ready and activated"
echo "   2. Install Ollama, download model and create extended model (see Start Guide - index.html)"
echo "   3. Start Jupyter: jupyter notebook notebooks/lesson_all_in_one.ipynb"