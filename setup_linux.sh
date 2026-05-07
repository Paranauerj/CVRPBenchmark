#!/bin/bash
# Installation script for Linux (Ubuntu/Debian)
set -e

echo "📦 Updating package list..."
sudo apt update

echo "🛠️ Installing build dependencies..."
# python3-dev is often needed to compile C++ extensions like hygese
sudo apt install -y build-essential cmake gcc g++ make python3-dev python3-pip python3-venv

echo "🐍 Setting up Python environment..."
# It's recommended to use a virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created."
fi

source venv/bin/activate

echo "📥 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo "------------------------------------------------"
echo "To start the application, run:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo "------------------------------------------------"
