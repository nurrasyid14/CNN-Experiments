@echo off
setlocal
cd /d "%~dp0.."

build\ingestion\Release\cnn_ingest.exe ^
  --root "data\raw\SIMPLIcity Dataset" ^
  --out "data\manifests\simplicity.jsonl" ^
  --label-mode numeric-stem --label-divisor 100
