"""Модель калорийности, seed, train/eval, конфиг (этап 2 ТЗ)."""

from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn as nn
import torchmetrics
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


@dataclass
class Config:
    """Конфиг обучения. Все запуски — через train(cfg)."""

    CONFIG: str = "full"
    SEED: int = 42
    DATA_ROOT: str = "data"
    IMAGE_SIZE: int = 160
    BATCH_SIZE: int = 4
    EPOCHS: int = 12
    PATIENCE: int = 3
    HIDDEN_DIM: int = 128
    MAX_LENGTH: int = 64
    TEXT_MODEL: str = "distilbert-base-uncased"
    IMAGE_MODEL: str = "efficientnet_b0"
    TEXT_LR: float = 2e-5
    IMAGE_LR: float = 2e-5
    CLASSIFIER_LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    FREEZE_TEXT: bool = False
    FREEZE_IMAGE: bool = False
    USE_AUG: bool = False
    CHECKPOINT_PATH: str = "artifacts/best.pt"
    DEVICE: str = "auto"
    NUM_WORKERS: int = 0
    VAL_FRACTION: float = 0.1
    NAME: str = "full"


def set_seed(seed: int = 42) -> None:
    """Воспроизводимость результатов."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultimodalCalorieModel(nn.Module):
    """Image encoder + text encoder + projections + head → 1 (регрессия)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = timm.create_model(
            cfg.IMAGE_MODEL, pretrained=True, num_classes=0, global_pool="avg"
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
            img_dim = self.image_encoder(dummy).shape[-1]
        self.text_encoder = AutoModel.from_pretrained(cfg.TEXT_MODEL)
        text_dim = self.text_encoder.config.hidden_size

        self.image_projection = nn.Sequential(
            nn.Linear(img_dim, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # ТЗ этап 3: один выходной нейрон без активации
        self.head = nn.Linear(cfg.HIDDEN_DIM * 2, 1)
        # ТЗ AdamW называет финальный слой classifier — тот же модуль
        self.classifier = self.head

        if cfg.FREEZE_IMAGE:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        if cfg.FREEZE_TEXT:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def forward(
        self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        img_feat = self.image_projection(self.image_encoder(image))
        txt_feat = self.text_projection(self.encode_text(input_ids, attention_mask))
        return self.head(torch.cat([img_feat, txt_feat], dim=-1)).squeeze(-1)


def build_optimizer(model: MultimodalCalorieModel, cfg: Config) -> AdamW:
    """AdamW: энкодеры, проекции и classifier с раздельными LR (как в ТЗ)."""
    groups = [
        {"params": model.text_encoder.parameters(), "lr": cfg.TEXT_LR},
        {"params": model.image_encoder.parameters(), "lr": cfg.IMAGE_LR},
        {"params": model.text_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
        {"params": model.image_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
        {"params": model.classifier.parameters(), "lr": cfg.CLASSIFIER_LR},
    ]
    groups = [g for g in groups if any(p.requires_grad for p in g["params"])]
    return AdamW(groups, weight_decay=cfg.WEIGHT_DECAY)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Средний MAE на loader (torchmetrics.MeanAbsoluteError)."""
    model.eval()
    metric = torchmetrics.MeanAbsoluteError().to(device)
    for batch in loader:
        image = batch["image"].to(device)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["calories"].to(device)
        pred = model(image, ids, mask)
        metric.update(pred, y)
    return float(metric.compute().item())


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _loaders_from_cfg(cfg: Config):
    from transformers import AutoTokenizer

    from .dataset import load_dish_table, make_loaders, split_frames

    dish = load_dish_table(cfg.DATA_ROOT)
    train_df, val_df, test_df = split_frames(dish, cfg.SEED, cfg.VAL_FRACTION)
    print("sizes train/val/test:", len(train_df), len(val_df), len(test_df))
    tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL)
    train_loader, val_loader, _test_loader = make_loaders(
        train_df,
        val_df,
        test_df,
        tokenizer,
        image_size=cfg.IMAGE_SIZE,
        batch_size=cfg.BATCH_SIZE,
        use_aug=cfg.USE_AUG,
        max_length=cfg.MAX_LENGTH,
        num_workers=cfg.NUM_WORKERS,
        data_root=cfg.DATA_ROOT,
    )
    return train_loader, val_loader, dish


def prepare_test_loader(cfg: Config):
    """Test loader для этапа 4 (инференс с чекпоинта)."""
    from transformers import AutoTokenizer

    from .dataset import (
        CaloriesDataset,
        collate_calories,
        get_eval_transforms,
        load_dish_table,
    )

    dish = load_dish_table(cfg.DATA_ROOT)
    test_df = dish[(dish["exists"]) & (dish["split"] == "test")].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL)
    ds = CaloriesDataset(
        test_df,
        get_eval_transforms(cfg.IMAGE_SIZE),
        tokenizer,
        cfg.MAX_LENGTH,
        data_root=cfg.DATA_ROOT,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_calories,
    )
    return loader, dish, test_df


def train(cfg: Config) -> dict[str, Any]:
    """Обучение по конфигу: loaders собираются внутри из cfg.DATA_ROOT."""
    set_seed(cfg.SEED)
    train_loader, val_loader, dish = _loaders_from_cfg(cfg)
    if cfg.DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg.DEVICE = str(device)
    else:
        device = torch.device(cfg.DEVICE)

    model = MultimodalCalorieModel(cfg).to(device)
    opt = build_optimizer(model, cfg)
    criterion = nn.L1Loss()
    history = {"train_mae": [], "val_mae": [], "epoch_sec": []}
    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0
    n_params = count_params(model)
    print(f"=== Train {cfg.NAME} | trainable_params={n_params:,} | device={device} ===")
    print(asdict(cfg))

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        t0 = time.perf_counter()
        run, n = 0.0, 0
        for batch in train_loader:
            image = batch["image"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["calories"].to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(image, ids, mask)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += torch.abs(pred.detach() - y).sum().item()
            n += y.numel()
        train_mae = run / max(1, n)
        val_mae = evaluate(model, val_loader, device)
        sec = time.perf_counter() - t0
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["epoch_sec"].append(sec)
        mark = ""
        if val_mae < best_val - 1e-4:
            best_val, best_epoch = val_mae, epoch
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
            mark = " *best*"
        else:
            bad += 1
        print(
            f"epoch {epoch}/{cfg.EPOCHS} | train_mae={train_mae:.2f} | "
            f"val_mae={val_mae:.2f} | {sec:.1f}s{mark}"
        )
        if bad >= cfg.PATIENCE:
            print(f"early stopping (best_epoch={best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_dir = os.path.dirname(cfg.CHECKPOINT_PATH)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "val_mae": best_val},
        cfg.CHECKPOINT_PATH,
    )
    Path(cfg.CHECKPOINT_PATH).with_suffix(".json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("checkpoint saved:", cfg.CHECKPOINT_PATH)
    print(f"DONE {cfg.NAME} | val_mae={best_val:.2f} | best_epoch={best_epoch}")
    return {
        "name": cfg.NAME,
        "val_mae": best_val,
        "best_epoch": best_epoch,
        "trainable_params": n_params,
        "sec_per_epoch": float(np.mean(history["epoch_sec"])) if history["epoch_sec"] else 0.0,
        "history": history,
        "model": model,
        "cfg": cfg,
        "device": device,
        "dish": dish,
    }


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device | None = None,
    allowed_dir: str | os.PathLike | None = None,
):
    """Загрузка модели из артефакта (этап 4). Только .pt внутри allowed_dir."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(checkpoint_path).expanduser().resolve()
    base = Path(allowed_dir).expanduser().resolve() if allowed_dir else path.parent
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Чекпоинт вне {base}: {path}") from exc
    if path.suffix != ".pt" or not path.is_file():
        raise FileNotFoundError(path)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError as exc:
        raise RuntimeError("Нужен PyTorch с weights_only=True, pickle-чекпоинт не читаем") from exc
    cfg_path = path.with_suffix(".json")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Рядом с чекпоинтом нужен JSON-конфиг: {cfg_path}")
    cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = Config(**{k: v for k, v in cfg_raw.items() if k in Config.__dataclass_fields__})
    model = MultimodalCalorieModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, {**ckpt, "cfg": cfg_raw}


# ТЗ требует этот импорт в utils.py; tokenize_fn удобно собирать через partial.
make_tokenizer = partial(AutoTokenizer.from_pretrained)
