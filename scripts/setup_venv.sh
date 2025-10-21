#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting environment setup..."
echo ""

# Detect Python command
if command -v python &> /dev/null; then
    PY="python"
elif command -v python3 &> /dev/null; then
    PY="python3"
else
    echo "❌ Error: Python not found. Please install Python 3.12+"
    exit 1
fi

echo "✅ Found Python: ${PY}"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
${PY} -m venv .venv

echo ""
echo "⬆️  Upgrading pip..."
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

python -m pip install --upgrade pip

# Install requirements
echo ""
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Copy .env.example to .env with Ollama as default
echo ""
echo "⚙️  Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created .env file with Ollama as default backend"
else
    echo "ℹ️  .env file already exists, skipping..."
fi

# Create Modelfile for extended token limit
echo ""
echo "📝 Creating Modelfile for Ollama (extended token limit)..."
if [ ! -f "Modelfile" ]; then
    cat > Modelfile << 'EOF'
# Based on llama3.2:3b with increased token limit
FROM llama3.2:3b

# Increase max tokens from default 128 to 300
PARAMETER num_predict 300
EOF
    echo "✅ Created Modelfile (will be used after Ollama is installed)"
else
    echo "ℹ️  Modelfile already exists, skipping..."
fi

echo ""
echo "✅ Setup complete!"