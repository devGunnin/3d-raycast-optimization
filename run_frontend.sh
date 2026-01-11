#!/bin/bash
# Run Tactica Frontend Server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "   Tactica Camera Placement Optimization   "
echo "============================================"
echo ""

# Check if virtual environment exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
elif [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# Check for required packages
python3 -c "import fastapi" 2>/dev/null || {
    echo "Installing required packages..."
    pip install fastapi uvicorn websockets
}

echo "Starting server..."
echo "Open http://localhost:6969 in your browser"
echo ""

cd "$SCRIPT_DIR"
python3 frontend/server.py
