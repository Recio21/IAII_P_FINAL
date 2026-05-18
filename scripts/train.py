"""
train.py — Pipeline de entrenamiento para el clasificador de emociones faciales.

Uso:
    python scripts/train.py [--epochs N] [--model-version N] [--data-dir PATH]

Salidas en models/:
    model_<version>_final.keras
    history_<version>.pkl
    metadata_<version>.json
    metrics.json              ← leído por el workflow de CI
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

SEED      = 42
IMG_SIZE  = (128, 128)
BATCH     = 32
CLASSES   = ["Angry", "Fear", "Happy", "Sad", "Surprise"]


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)
    except ImportError:
        pass


def download_dataset() -> Path:
    """Descarga el dataset desde Kaggle Hub y devuelve la ruta a /Data."""
    try:
        import kagglehub
        base = Path(kagglehub.dataset_download("samithsachidanandan/human-face-emotions"))
        data_dir = base / "Data"
        if data_dir.exists():
            return data_dir
        # Algunos downloads colocan los datos un nivel más arriba
        candidates = list(base.rglob("Data"))
        if candidates:
            return candidates[0]
        return base
    except Exception as exc:
        print(f"[ERROR] kagglehub download failed: {exc}", file=sys.stderr)
        sys.exit(1)


def build_datasets(data_dir: Path):
    import tensorflow as tf
    load_kw = dict(
        validation_split=0.2,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        color_mode="grayscale",
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir), subset="training", **load_kw
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir), subset="validation", **load_kw
    )
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE)
    val_ds   = val_ds.map(  lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds


def build_model(num_classes: int):
    import tensorflow as tf

    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
    ], name="augmentation")

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 1))
    x = aug(inputs)

    for filters in (32, 64, 128, 256):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(data_dir: Path, epochs: int, version: int):
    import tensorflow as tf

    set_seeds()
    print(f"[train] data_dir={data_dir}  epochs={epochs}  version={version}")

    train_ds, val_ds = build_datasets(data_dir)
    model = build_model(len(CLASSES))
    model.summary(print_fn=lambda s: print(f"  {s}"))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    train_seconds = round(time.time() - t0, 1)

    # ── Métricas finales ──────────────────────────────────────────────────
    best_epoch      = int(np.argmin(history.history["val_loss"]))
    val_acc         = float(history.history["val_accuracy"][best_epoch])
    val_loss        = float(history.history["val_loss"][best_epoch])
    train_acc       = float(history.history["accuracy"][best_epoch])
    epochs_run      = len(history.history["val_loss"])

    # ── Guardar artefactos ────────────────────────────────────────────────
    model_name    = f"model_{version}_final.keras"
    history_name  = f"history_{version}.pkl"
    meta_name     = f"metadata_{version}.json"

    model_path   = MODELS_DIR / model_name
    history_path = MODELS_DIR / history_name
    meta_path    = MODELS_DIR / meta_name

    model.save(str(model_path))
    print(f"[train] model saved → {model_path}")

    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)

    metadata = {
        "class_names": CLASSES,
        "IMG_SIZE":    list(IMG_SIZE),
        "BATCH_SIZE":  BATCH,
        "model_version": f"model_{version}_final",
        "epochs_trained": epochs_run,
        "best_epoch":  best_epoch + 1,
        "train_accuracy": round(train_acc, 4),
        "val_accuracy":   round(val_acc, 4),
        "val_loss":       round(val_loss, 4),
        "train_seconds":  train_seconds,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    # metrics.json — leído por el workflow de CI
    metrics = {
        "version":        version,
        "model_file":     model_name,
        "metadata_file":  meta_name,
        "val_accuracy":   round(val_acc, 4),
        "val_loss":       round(val_loss, 4),
        "train_accuracy": round(train_acc, 4),
        "epochs_trained": epochs_run,
        "best_epoch":     best_epoch + 1,
        "model_size_mb":  round(model_path.stat().st_size / 1024 / 1024, 2),
        "train_seconds":  train_seconds,
    }
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\n[train] ── Results ──────────────────────────────")
    print(f"  val_accuracy  : {val_acc:.4f}")
    print(f"  val_loss      : {val_loss:.4f}")
    print(f"  epochs_trained: {epochs_run}")
    print(f"  model_size_mb : {metrics['model_size_mb']}")
    print(f"  train_seconds : {train_seconds}")
    print(f"  metrics.json  → {metrics_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int, default=30)
    parser.add_argument("--model-version", type=int, default=None,
                        help="Número de versión del modelo (auto-detectado si no se indica)")
    parser.add_argument("--data-dir",      type=str, default=None,
                        help="Ruta al directorio /Data del dataset (descarga automática si no se indica)")
    args = parser.parse_args()

    # Auto-detect next version
    if args.model_version is None:
        existing = sorted(MODELS_DIR.glob("model_*_final.keras"))
        if existing:
            last = int(existing[-1].name.split("_")[1])
            args.model_version = last + 1
        else:
            args.model_version = 1

    data_dir = Path(args.data_dir) if args.data_dir else download_dataset()
    train(data_dir, args.epochs, args.model_version)
