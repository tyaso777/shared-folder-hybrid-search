param(
    [string]$ModelId = "cl-nagoya/ruri-v3-310m",
    [string]$OutputDir = "models\ruri-v3-onnx",
    [string]$PythonVersion = "3.11",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path (Get-Location) $Path)
}

$output = Resolve-RepoPath $OutputDir
$venv = Resolve-RepoPath ".venv-onnx"
$requirements = Resolve-RepoPath "scripts\onnx-export-requirements.txt"
$python = Join-Path $venv "Scripts\python.exe"
$optimum = Join-Path $venv "Scripts\optimum-cli.exe"

if ((Test-Path $output) -and -not $Force) {
    Write-Host "Output directory already exists: $output"
    Write-Host "Use -Force to delete and recreate it."
    exit 1
}

if (Test-Path $output) {
    Remove-Item -Recurse -Force $output
}
New-Item -ItemType Directory -Force -Path $output | Out-Null

if (-not (Test-Path $python)) {
    uv python install $PythonVersion
    uv venv $venv --python $PythonVersion
}

uv pip install --python $python -r $requirements

& $optimum export onnx `
    --model $ModelId `
    --task feature-extraction `
    $output

$dllPath = & $python -c "import pathlib, onnxruntime as ort; p=pathlib.Path(ort.__file__).parent/'capi'/'onnxruntime.dll'; print(p)"
if (-not (Test-Path $dllPath)) {
    throw "onnxruntime.dll was not found at $dllPath"
}

Copy-Item $dllPath (Join-Path $output "onnxruntime.dll") -Force

$model = Join-Path $output "model.onnx"
$tokenizer = Join-Path $output "tokenizer.json"
$runtime = Join-Path $output "onnxruntime.dll"

if (-not (Test-Path $model)) { throw "model.onnx was not generated" }
if (-not (Test-Path $tokenizer)) { throw "tokenizer.json was not generated" }
if (-not (Test-Path $runtime)) { throw "onnxruntime.dll was not copied" }

Write-Host ""
Write-Host "Export complete."
Write-Host "Model:     $model"
Write-Host "Tokenizer: $tokenizer"
Write-Host "Runtime:   $runtime"
Write-Host ""
Write-Host "Build vector index example:"
Write-Host "cargo run -p build-index -- --dataset jawikibooks --schema examples\jawikibooks\schema.json --input examples\jawikibooks\input.jsonl --indexes-root indexes --embedding-model `"$model`" --tokenizer `"$tokenizer`" --ort-dll `"$runtime`" --embedding-dim 768"
