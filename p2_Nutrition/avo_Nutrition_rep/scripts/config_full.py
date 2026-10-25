"""Конфиг полного обучения (этап 2 ТЗ: конфиг-файл)."""

from __future__ import annotations

from .utils import Config


def get_full_config(data_root: str, checkpoint_path: str) -> Config:
    """Конфигурация сдаточного / GPU-прогона (C7: 160px, без aug, BATCH=4)."""
    return Config(
        CONFIG="full",
        NAME="full_c7_effb0",
        SEED=42,
        DATA_ROOT=data_root,
        IMAGE_SIZE=160,
        BATCH_SIZE=4,
        EPOCHS=12,
        PATIENCE=3,
        HIDDEN_DIM=128,
        MAX_LENGTH=64,
        TEXT_MODEL="distilbert-base-uncased",
        IMAGE_MODEL="efficientnet_b0",
        TEXT_LR=2e-5,
        IMAGE_LR=2e-5,
        CLASSIFIER_LR=1e-3,
        WEIGHT_DECAY=1e-4,
        FREEZE_TEXT=False,
        FREEZE_IMAGE=False,
        USE_AUG=False,
        CHECKPOINT_PATH=checkpoint_path,
        DEVICE="auto",
        NUM_WORKERS=0,
        VAL_FRACTION=0.1,
    )
