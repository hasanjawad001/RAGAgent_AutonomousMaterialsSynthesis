#!/bin/bash
# Make uv use the OS trust store (fixes TLS behind enterprise proxies)
export UV_NATIVE_TLS=true

# === Ensure ~/.local/bin is in PATH ===
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo "[INFO] Adding ~/.local/bin to PATH for this session..."
    export PATH="$HOME/.local/bin:$PATH"
fi

# === Ensure uv is installed ===
if ! command -v uv &> /dev/null; then
    echo "[INFO] uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # After installation, add ~/.local/bin to PATH again (for safety)
    export PATH="$HOME/.local/bin:$PATH"

    # Try sourcing the env file if it exists
    if [ -f "$HOME/.local/bin/env" ]; then
        echo "[INFO] Sourcing uv environment..."
        source "$HOME/.local/bin/env"
    fi

    # Verify uv installation
    if ! command -v uv &> /dev/null; then
        echo "[⚠️] uv was installed but not detected in PATH."
        echo "[⚠️] Please restart your terminal and rerun this script."
        exit 1
    fi
fi

# === Create virtual environment ===
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment..."
    uv venv --python 3.12
fi

# === Activate virtual environment ===
source .venv/bin/activate

# === Install dependencies ===
echo "[INFO] Installing dependencies..."
uv pip install -r requirements.txt

# === Run the app ===
echo "[INFO] Starting Streamlit app..."
uv run streamlit run app.py
