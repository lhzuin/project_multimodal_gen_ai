# run_uci.ps1
$ProjectRoot = "E:\Ariel\project_multimodal_gen_ai"
$PythonExe   = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$Ckpt        = Join-Path $ProjectRoot "checkpoints\chess_llm_decoder_v3.pt"
$Tokenizer   = Join-Path $ProjectRoot "tokenizers\chess_uci_vocab.json"

Set-Location $ProjectRoot

& $PythonExe -u -m utils.uci_engine `
  --model_type decoder `
  --ckpt $Ckpt `
  --tokenizer_path $Tokenizer