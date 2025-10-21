#!/bin/bash

# Make this script executable automatically
chmod +x "$0" 2>/dev/null

echo "🚀 Starting Jupyter Notebook..."
echo "📖 Opening lesson_all_in_one.ipynb in your browser..."
echo ""
echo "Press Ctrl+C to stop Jupyter when you're done."
echo ""

# Start Jupyter and open notebook directly
jupyter notebook notebooks/lesson_all_in_one.ipynb