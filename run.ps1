$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "未找到 .venv。请先按 README 执行 python -m venv .venv 并安装 requirements.txt。"
}

Set-Location -LiteralPath $projectRoot
& $pythonPath -m streamlit run app.py `
    --server.address 127.0.0.1 `
    --server.port 8501 `
    --server.headless true `
    --browser.gatherUsageStats false
