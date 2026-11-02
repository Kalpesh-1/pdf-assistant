# PDF Assistant - one-command setup for Windows.
#
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#
# Installs uv if missing, installs Python and all packages, and creates .env.
# Safe to run again: it never overwrites an existing .env, so your API key
# cannot be destroyed by re-running this.

$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

Write-Host "=============================================="
Write-Host " PDF Assistant - setup"
Write-Host "=============================================="
Write-Host ""

# --- 1. uv -------------------------------------------------------------------
# uv installs itself to %USERPROFILE%\.local\bin, which is usually NOT on PATH
# in the session that is already running, so we add it here as well as checking.
$uvBin = Join-Path $env:USERPROFILE ".local\bin"
$env:PATH = "$uvBin;$env:PATH"

if (Get-Command uv -ErrorAction SilentlyContinue) {
    $version = (uv --version)
    Write-Host "[1/3] uv is already installed ($version)"
} else {
    Write-Host "[1/3] Installing uv..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:PATH = "$uvBin;$env:PATH"

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host ""
        Write-Host "ERROR: uv was installed but is not on PATH in this session." -ForegroundColor Red
        Write-Host "Close this window, open a new PowerShell, and run this script again."
        exit 1
    }
    $version = (uv --version)
    Write-Host "      done ($version)"
}

# --- 2. Python and packages --------------------------------------------------
# uv reads .python-version and uv.lock, so it fetches Python 3.12 if the machine
# does not have it and installs the exact package versions this project was
# built against.
Write-Host ""
Write-Host "[2/3] Installing Python and packages (about 200 MB, a few minutes)..."

uv sync
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "The download failed. This is usually a slow or unreliable connection."
    Write-Host "Retrying one package at a time, which is slower but far more reliable..."
    $env:UV_HTTP_TIMEOUT = "1800"
    $env:UV_CONCURRENT_DOWNLOADS = "1"
    uv sync
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Install failed again. See the messages above." -ForegroundColor Red
        exit 1
    }
}
Write-Host "      done"

# --- 3. Configuration file ---------------------------------------------------
Write-Host ""
$needsKey = $false

if (Test-Path .env) {
    Write-Host "[3/3] .env already exists - leaving it untouched"
} else {
    Copy-Item .env.example .env
    Write-Host "[3/3] Created .env from the template"
    $needsKey = $true
}

if (-not $needsKey) {
    if (Select-String -Path .env -Pattern '^GROQ_API_KEY=your_groq_api_key' -Quiet) {
        $needsKey = $true
    }
}

# --- What next ---------------------------------------------------------------
Write-Host ""
Write-Host "=============================================="
Write-Host " Setup complete"
Write-Host "=============================================="

if ($needsKey) {
    Write-Host ""
    Write-Host "ONE MANUAL STEP LEFT - add your Groq API key:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  1. Get a free key at https://console.groq.com/keys"
    Write-Host "  2. Open the file:  $(Join-Path (Get-Location) '.env')"
    Write-Host "  3. Replace   GROQ_API_KEY=your_groq_api_key"
    Write-Host "     with      GROQ_API_KEY=<your real key>"
    Write-Host ""
    Write-Host "Without the key you can still upload and read PDFs, but the app"
    Write-Host "cannot answer questions or write summaries."
}

Write-Host ""
Write-Host "Then start the app with:"
Write-Host ""
Write-Host "  uv run streamlit run app.py"
Write-Host ""
Write-Host "It opens at http://localhost:8501"
Write-Host ""
