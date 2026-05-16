# PipelineModeling

Sistema MLOps local para servir una CNN de clasificación de emociones faciales (`model_7_final.keras`).

## Arranque

```bash
cp .env.example .env
docker compose up -d --build
```

| Servicio   | URL                       |
|------------|---------------------------|
| Backend    | http://localhost:8000     |
| Frontend   | http://localhost:8080     |
| Prometheus | http://localhost:9090     |
| Grafana    | http://localhost:3000     |

## Endpoints

| Método | Ruta              | Descripción                          |
|--------|-------------------|--------------------------------------|
| GET    | `/health`         | Healthcheck                          |
| GET    | `/model/status`   | Versión, clases, tamaño, canales     |
| POST   | `/predict`        | `multipart/form-data` `file=image`   |
| POST   | `/model/switch`   | `{"model_file":"otro.keras"}`        |
| POST   | `/retrain`        | Trigger placeholder + `dvc pull`     |
| POST   | `/observations`   | Almacena observación en JSONL        |
| GET    | `/metrics`        | Prometheus                           |

## Coloca tus adjuntos

```
PipelineModeling/models/model_7_final.keras
PipelineModeling/models/metadata_7.json
PipelineModeling/models/history_7.pkl
PipelineModeling/notebooks/Untitled0.ipynb
```

## DVC

```bash
cd PipelineModeling
bash scripts/dvc_init.sh
# Remoto opcional:
dvc remote add -d storage s3://my-bucket/pipelinemodeling
dvc push
```
