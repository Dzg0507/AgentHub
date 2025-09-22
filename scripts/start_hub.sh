#!/bin/bash

echo "🤖 Starting Agent Control Hub..."
echo "================================"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python is not installed or not in PATH"
    exit 1
fi

# Make the script executable
chmod +x "$0"

# Start the unified hub
python start_hub.py
