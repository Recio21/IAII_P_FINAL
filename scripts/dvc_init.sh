#!/usr/bin/env bash
# Inicialización DVC para artefactos del modelo. Ejecutar desde PipelineModeling/.
set -euo pipefail

if ! command -v dvc >/dev/null 2>&1; then
  echo "DVC not installed. pip install dvc" >&2
  exit 1
fi

dvc init -q || true

dvc add models/model_7_final.keras
dvc add models/metadata_7.json
dvc add models/history_7.pkl

git add models/*.dvc models/.gitignore .gitignore
git commit -m "Track trained emotion CNN model with DVC" || true

# Para remoto opcional (S3 / GCS / Azure / local):
#   dvc remote add -d storage s3://your-bucket/pipelinemodeling
#   dvc push
echo "DVC ready. Configure a remote with: dvc remote add -d storage <url>"
