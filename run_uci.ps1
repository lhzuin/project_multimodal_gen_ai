# Register this script as a UCI engine; use environment variables for local paths.
$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$PythonExe = if ($env:PYTHON_EXECUTABLE) { $env:PYTHON_EXECUTABLE } else { Join-Path $ProjectRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path $PythonExe)) { $PythonExe = "python" }
$ModelType = if ($env:MODEL_TYPE) { $env:MODEL_TYPE } else { "decoder" }
$Checkpoint = if ($env:CHECKPOINT) { $env:CHECKPOINT } else { Join-Path $ProjectRoot "checkpoints\chess_llm_decoder_v12.pt" }
$Layers = if ($env:N_LAYERS) { $env:N_LAYERS } else { "12" }
Set-Location $ProjectRoot
& $PythonExe -u -m utils.uci_engine --model_type $ModelType --ckpt $Checkpoint --n_layers $Layers --tokenizer_path (Join-Path $ProjectRoot "tokenizers\chess_uci_vocab.json") @args
exit $LASTEXITCODE
