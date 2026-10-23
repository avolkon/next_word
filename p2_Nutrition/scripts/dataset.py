"""Датасет блюд: изображение rgb.png + текст ингредиентов → калории.

Путь к фото (ТЗ):
    img_path = os.path.join(data_root, "images", str(dish_id), "rgb.png")
    image = Image.open(img_path).convert("RGB")
"""

from __future__ import annotations

import os
import re
from typing import Optional

import albumentations as A
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def build_ingredient_map(ingredients_csv: str | os.PathLike) -> dict[int, str]:
    """id (int) → название ингредиента."""
    df = pd.read_csv(ingredients_csv)
    return {int(r["id"]): str(r["ingr"]).strip() for _, r in df.iterrows()}


def parse_ingredient_ids(raw: str) -> list[int]:
    """'ingr_0000000122;ingr_0000000026' → [122, 26]."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    ids = []
    for part in raw.split(";"):
        part = part.strip()
        m = re.search(r"(\d+)$", part)
        if m:
            ids.append(int(m.group(1)))
    return ids


def ingredients_to_text(raw: str, id2name: dict[int, str]) -> str:
    """Технические id → строка названий через запятую."""
    names = []
    for i in parse_ingredient_ids(raw):
        name = id2name.get(i)
        if name:
            names.append(name)
    return ", ".join(names) if names else "unknown"


def load_dish_table(data_root: str | os.PathLike) -> pd.DataFrame:
    """dish.csv + текстовые ингредиенты + путь к rgb.png."""
    data_root = os.fspath(data_root)
    dish = pd.read_csv(os.path.join(data_root, "dish.csv"))
    id2name = build_ingredient_map(os.path.join(data_root, "ingredients.csv"))
    dish = dish.copy()
    dish["ingredients_text"] = dish["ingredients"].map(
        lambda x: ingredients_to_text(x, id2name)
    )
    dish["image_path"] = dish["dish_id"].map(
        lambda d: os.path.join(data_root, "images", str(d), "rgb.png")
    )
    dish["exists"] = dish["image_path"].map(os.path.isfile)
    return dish


def get_train_transforms(image_size: int, use_aug: bool = True) -> A.Compose:
    """Аугментации только для train."""
    ops = [A.Resize(image_size, image_size)]
    if use_aug:
        ops.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
            ]
        )
    ops.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(ops)


def get_eval_transforms(image_size: int) -> A.Compose:
    """Детерминированные преобразования для val/test."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class CaloriesDataset(Dataset):
    """Один пример: image tensor, ingredients text, calories float."""

    def __init__(
        self,
        frame: pd.DataFrame,
        transform: A.Compose,
        tokenizer=None,
        max_length: int = 64,
    ):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image_np = np.array(image)
        image_t = self.transform(image=image_np)["image"]
        text = str(row["ingredients_text"])
        calories = float(row["total_calories"])
        item = {
            "image": image_t,
            "text": text,
            "calories": calories,
            "dish_id": str(row["dish_id"]),
            "mass": float(row["total_mass"]),
        }
        if self.tokenizer is not None:
            enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            item["input_ids"] = enc["input_ids"].squeeze(0)
            item["attention_mask"] = enc["attention_mask"].squeeze(0)
        return item


def collate_calories(batch: list[dict]) -> dict:
    """Склеивает батч для модели с токенизатором."""
    import torch

    images = torch.stack([b["image"] for b in batch], dim=0)
    calories = torch.tensor([b["calories"] for b in batch], dtype=torch.float32)
    out = {
        "image": images,
        "calories": calories,
        "text": [b["text"] for b in batch],
        "dish_id": [b["dish_id"] for b in batch],
        "mass": torch.tensor([b["mass"] for b in batch], dtype=torch.float32),
    }
    if "input_ids" in batch[0]:
        out["input_ids"] = torch.stack([b["input_ids"] for b in batch], dim=0)
        out["attention_mask"] = torch.stack([b["attention_mask"] for b in batch], dim=0)
    return out


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    image_size: int,
    batch_size: int,
    use_aug: bool,
    max_length: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Три DataLoader для train/val/test."""
    train_ds = CaloriesDataset(
        train_df, get_train_transforms(image_size, use_aug), tokenizer, max_length
    )
    val_ds = CaloriesDataset(
        val_df, get_eval_transforms(image_size), tokenizer, max_length
    )
    test_ds = CaloriesDataset(
        test_df, get_eval_transforms(image_size), tokenizer, max_length
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_calories,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_calories,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_calories,
    )
    return train_loader, val_loader, test_loader
