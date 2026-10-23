"""Пайплайн Goods: аудит, transforms, словарь, Dataset, DataLoader, 6 конфигов."""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class PipelineCfg:
    NAME: str = "G1"
    SEED: int = 42
    IMAGE_SIZE: int = 128
    MAX_TEXT_LEN: int = 16
    BATCH_SIZE: int = 8
    STRONG_AUG: bool = False
    IMAGENET_NORM: bool = True
    MIN_FREQ: int = 1


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return clean_text(text).split()


def build_vocab(texts: list[str], min_freq: int = 1) -> tuple[dict[str, int], dict[int, str]]:
    counts: Counter[str] = Counter()
    for t in texts:
        counts.update(tokenize(t))
    specials = ["<PAD>", "<UNK>"]
    vocab = {tok: i for i, tok in enumerate(specials)}
    for tok, c in sorted(counts.items()):
        if c >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    inv = {i: t for t, i in vocab.items()}
    return vocab, inv


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokenize(text)]
    ids = ids[:max_len]
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids


def get_transforms(cfg: PipelineCfg, train: bool) -> transforms.Compose:
    size = cfg.IMAGE_SIZE
    ops: list[Any] = []
    if train:
        if cfg.STRONG_AUG:
            ops.extend(
                [
                    transforms.Resize((size + 32, size + 32)),
                    transforms.RandomCrop(size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
                ]
            )
        else:
            ops.extend(
                [
                    transforms.Resize((size, size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.15, 0.15, 0.1),
                ]
            )
    else:
        ops.append(transforms.Resize((size, size)))
    ops.append(transforms.ToTensor())
    if cfg.IMAGENET_NORM:
        ops.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(ops)


class FashionDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        frame: pd.DataFrame,
        image_tf: transforms.Compose,
        vocab: dict[str, int],
        max_len: int,
    ):
        self.root = Path(root)
        self.frame = frame.reset_index(drop=True)
        self.image_tf = image_tf
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]
        path = self.root / row["image_path"]
        image = Image.open(path).convert("RGB")
        image_t = self.image_tf(image)
        name = str(row["productDisplayName"])
        text_ids = torch.tensor(encode_text(name, self.vocab, self.max_len), dtype=torch.long)
        label = int(row["label"])
        return {
            "image": image_t,
            "text": text_ids,
            "label": label,
            "image_path": str(row["image_path"]),
            "product_name": name,
            "class_name": str(row["class_name"]),
        }


def audit_dataset(root: Path) -> dict[str, Any]:
    meta = pd.read_csv(root / "metadata.csv")
    c2i = pd.read_csv(root / "class_to_idx.csv")
    report: dict[str, Any] = {
        "n_rows": len(meta),
        "columns": list(meta.columns),
        "split_counts": meta["split"].value_counts().to_dict(),
        "class_counts": {
            f"{a}|{b}": int(v)
            for (a, b), v in meta.groupby(["split", "class_name"]).size().items()
        },
        "missing_files": 0,
        "bad_images": 0,
        "empty_names": int((meta["productDisplayName"].fillna("").astype(str).str.strip() == "").sum()),
        "c2i_ok": True,
    }
    # class_to_idx consistency
    mapping = dict(zip(c2i["class_name"], c2i["label"]))
    bad_map = meta.apply(lambda r: mapping.get(r["class_name"]) != r["label"], axis=1).sum()
    report["label_mismatch"] = int(bad_map)
    report["c2i_ok"] = bad_map == 0

    widths, heights = [], []
    for _, row in meta.iterrows():
        p = root / row["image_path"]
        if not p.is_file():
            report["missing_files"] += 1
            continue
        try:
            im = Image.open(p)
            im.verify()
            im = Image.open(p)
            w, h = im.size
            widths.append(w)
            heights.append(h)
            if im.mode not in ("RGB", "RGBA", "L"):
                report["bad_images"] += 1
        except Exception:
            report["bad_images"] += 1
    report["img_w_mean"] = float(np.mean(widths)) if widths else None
    report["img_h_mean"] = float(np.mean(heights)) if heights else None
    report["ok"] = (
        report["missing_files"] == 0
        and report["bad_images"] == 0
        and report["label_mismatch"] == 0
        and report["empty_names"] == 0
    )
    return report, meta, c2i


def run_config(root: Path, meta: pd.DataFrame, cfg: PipelineCfg) -> dict[str, Any]:
    set_seed(cfg.SEED)
    train_df = meta[meta["split"] == "train"].copy()
    val_df = meta[meta["split"] == "val"].copy()
    test_df = meta[meta["split"] == "test"].copy()
    vocab, _ = build_vocab(train_df["productDisplayName"].tolist(), cfg.MIN_FREQ)

    train_ds = FashionDataset(root, train_df, get_transforms(cfg, True), vocab, cfg.MAX_TEXT_LEN)
    val_ds = FashionDataset(root, val_df, get_transforms(cfg, False), vocab, cfg.MAX_TEXT_LEN)
    test_ds = FashionDataset(root, test_df, get_transforms(cfg, False), vocab, cfg.MAX_TEXT_LEN)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    batch = next(iter(train_loader))
    shapes_ok = (
        tuple(batch["image"].shape) == (cfg.BATCH_SIZE, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
        and tuple(batch["text"].shape) == (cfg.BATCH_SIZE, cfg.MAX_TEXT_LEN)
        and tuple(batch["label"].shape) == (cfg.BATCH_SIZE,)
    )
    # sample one item check
    item = train_ds[0]
    item_ok = item["image"].ndim == 3 and item["text"].numel() == cfg.MAX_TEXT_LEN

    # denorm preview score: mean pixel in [0,1] range roughly if no crazy aug
    img = batch["image"][0]
    if cfg.IMAGENET_NORM:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        vis = (img * std + mean).clamp(0, 1)
    else:
        vis = img.clamp(0, 1)
    recognizable = float(vis.mean()) > 0.05  # not blank

    score = 0
    score += 10 if shapes_ok else 0
    score += 5 if item_ok else 0
    score += 3 if recognizable else 0
    score += 2 if len(vocab) > 50 else 0
    # prefer standard ImageNet + 224 for transfer learning readiness
    if cfg.IMAGE_SIZE == 224:
        score += 2
    if cfg.IMAGENET_NORM:
        score += 1
    if not cfg.STRONG_AUG:
        score += 1  # weak aug safer for product photos
    if cfg.MAX_TEXT_LEN >= 16:
        score += 1

    return {
        "config": cfg.NAME,
        "vocab_size": len(vocab),
        "batch_image_shape": list(batch["image"].shape),
        "batch_text_shape": list(batch["text"].shape),
        "batch_label_shape": list(batch["label"].shape),
        "shapes_ok": shapes_ok,
        "item_ok": item_ok,
        "recognizable": recognizable,
        "score": score,
        "image_size": cfg.IMAGE_SIZE,
        "max_text_len": cfg.MAX_TEXT_LEN,
        "batch_size": cfg.BATCH_SIZE,
        "strong_aug": cfg.STRONG_AUG,
        "imagenet_norm": cfg.IMAGENET_NORM,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "sample_name": item["product_name"],
        "sample_tokens": tokenize(item["product_name"])[: cfg.MAX_TEXT_LEN],
        "sample_ids": encode_text(item["product_name"], vocab, cfg.MAX_TEXT_LEN),
    }


CONFIGS = [
    PipelineCfg(NAME="G1", IMAGE_SIZE=128, MAX_TEXT_LEN=16, BATCH_SIZE=8, STRONG_AUG=False, IMAGENET_NORM=True),
    PipelineCfg(NAME="G2", IMAGE_SIZE=224, MAX_TEXT_LEN=16, BATCH_SIZE=8, STRONG_AUG=False, IMAGENET_NORM=True),
    PipelineCfg(NAME="G3", IMAGE_SIZE=224, MAX_TEXT_LEN=24, BATCH_SIZE=8, STRONG_AUG=False, IMAGENET_NORM=True),
    PipelineCfg(NAME="G4", IMAGE_SIZE=224, MAX_TEXT_LEN=16, BATCH_SIZE=8, STRONG_AUG=True, IMAGENET_NORM=True),
    PipelineCfg(NAME="G5", IMAGE_SIZE=224, MAX_TEXT_LEN=16, BATCH_SIZE=16, STRONG_AUG=False, IMAGENET_NORM=True),
    PipelineCfg(NAME="G6", IMAGE_SIZE=224, MAX_TEXT_LEN=16, BATCH_SIZE=8, STRONG_AUG=False, IMAGENET_NORM=False),
]


def main():
    root = Path(__file__).resolve().parents[1] / "Разработка" / "data_goods" / "fashion_project_dataset"
    out = Path(__file__).resolve().parents[1] / "Разработка" / "artifacts_cpu"
    out.mkdir(parents=True, exist_ok=True)

    report, meta, c2i = audit_dataset(root)
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("AUDIT ok=", report["ok"], "missing=", report["missing_files"], "bad_img=", report["bad_images"])
    print("splits", report["split_counts"])

    rows = []
    for cfg in CONFIGS:
        print("===", cfg.NAME, asdict(cfg))
        res = run_config(root, meta, cfg)
        rows.append(res)
        print("score", res["score"], "shapes", res["batch_image_shape"], res["batch_text_shape"], res["shapes_ok"])

    summary = pd.DataFrame(rows).sort_values(["score", "image_size", "max_text_len"], ascending=[False, False, False])
    summary.to_csv(out / "summary.csv", index=False)
    best = summary.iloc[0].to_dict()
    (out / "best_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSUMMARY")
    print(summary[["config", "score", "image_size", "max_text_len", "batch_size", "strong_aug", "imagenet_norm", "shapes_ok"]].to_string(index=False))
    print("BEST", best["config"], "score", best["score"])


if __name__ == "__main__":
    main()
