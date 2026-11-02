#!/usr/bin/env bash
#
# PDF Assistant - one-command setup for Linux and macOS.
#
#   bash setup.sh
#
# Installs uv if missing, installs Python and all packages, and creates .env.
# Safe to run again: it never overwrites an existing .env, so your API key
# cannot be destroyed by re-running this.

set -euo pipefail

cd "$(dirname "$0")"

echo "=============================================="
echo " PDF Assistant - setup"
echo "=============================================="
echo

# --- 1. uv -------------------------------------------------------------------
# uv installs itself to ~/.local/bin, which is usually NOT on PATH in the shell
# that is already running, so we add it here as well as checking for it.
export PATH="$HOME/.local/bin:$PATH"

if command -v uv >/dev/null 2>&1; then
    echo "[1/3] uv is already installed ($(uv --version))"
else
    echo "[1/3] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        echo
        echo "ERROR: uv was installed but is not on PATH in this shell."
        echo "Close this terminal, open a new one, and run 'bash setup.sh' again."
        exit 1
    fi
    echo "      done ($(uv --version))"
fi

# --- 2. Python and packages --------------------------------------------------
# uv reads .python-version and uv.lock, so it fetches Python 3.12 if the machine
# does not have it and installs the exact package versions this project was
# built against.
echo
echo "[2/3] Installing Python and packages (about 200 MB, a few minutes)..."

if ! uv sync; then
    echo
    echo "The download failed. This is usually a slow or unreliable connection."
    echo "Retrying one package at a time, which is slower but far more reliable..."
    UV_HTTP_TIMEOUT=1800 UV_CONCURRENT_DOWNLOADS=1 uv sync
fi
echo "      done"

# --- 3. Configuration file ---------------------------------------------------
echo
if [ -f .env ]; then
    echo "[3/3] .env already exists - leaving it untouched"
    NEEDS_KEY=0
else
    cp .env.example .env
    echo "[3/3] Created .env from the template"
    NEEDS_KEY=1
fi

# --- What next ---------------------------------------------------------------
echo
echo "=============================================="
echo " Setup complete"
echo "=============================================="

if [ "$NEEDS_KEY" = "1" ] || grep -q '^GROQ_API_KEY=your_groq_api_key' .env 2>/dev/null; then
    echo
    echo "ONE MANUAL STEP LEFT - add your Groq API key:"
    echo
    echo "  1. Get a free key at https://console.groq.com/keys"
    echo "  2. Open the file:  $(pwd)/.env"
    echo "  3. Replace   GROQ_API_KEY=your_groq_api_key"
    echo "     with      GROQ_API_KEY=<your real key>"
    echo
    echo "Without the key you can still upload and read PDFs, but the app"
    echo "cannot answer questions or write summaries."
fi

echo
echo "Then start the app with:"
echo
echo "  uv run streamlit run app.py"
echo
echo "It opens at http://localhost:8501"
echo
