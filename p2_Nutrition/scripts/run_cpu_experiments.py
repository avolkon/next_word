"""CPU: ≥6 конфигураций на малом срезе → summary.csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dataset import load_dish_table, make_loaders  # noqa: E402
from utils import Config, set_seed, train  # noqa: E402

DATA_ROOT = ROOT / "Разработка" / "data_nutrition"
OUT_DIR = ROOT / "Разработка" / "artifacts_cpu"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def split_frames(dish: pd.DataFrame, cfg: Config):
    set_seed(cfg.SEED)
    train_full = dish[dish["split"] == "train"].query("exists").copy()
    test_full = dish[dish["split"] == "test"].query("exists").copy()
    train_sub = train_full.sample(
        n=min(cfg.TRAIN_N + cfg.VAL_N, len(train_full)), random_state=cfg.SEED
    )
    va = train_sub.sample(n=min(cfg.VAL_N, len(train_sub)), random_state=cfg.SEED)
    tr = train_sub.drop(va.index)
    te = test_full.sample(n=min(cfg.TEST_N, len(test_full)), random_state=cfg.SEED)
    return tr, va, te


def base_cfg(**kwargs) -> Config:
    c = Config(
        CONFIG="cpu_small",
        DATA_ROOT=str(DATA_ROOT),
        DEVICE="cpu",
        EPOCHS=4,
        BATCH_SIZE=2,
        CHECKPOINT_PATH=str(OUT_DIR / "tmp.pt"),
    )
    for k, v in kwargs.items():
        setattr(c, k, v)
    c.CHECKPOINT_PATH = str(OUT_DIR / f"{c.NAME}.pt")
    return c


CONFIGS = [
    base_cfg(NAME="C1_baseline"),
    base_cfg(NAME="C2_text_unfreeze", FREEZE_TEXT=True, UNFREEZE_TEXT_LAST=True, TEXT_LR=5e-5),
    base_cfg(NAME="C3_image_unfreeze", FREEZE_IMAGE=True, UNFREEZE_IMAGE_LAST=True, IMAGE_LR=5e-5),
    base_cfg(NAME="C4_no_aug", USE_AUG=False),
    base_cfg(NAME="C5_size160", IMAGE_SIZE=160),
    base_cfg(NAME="C6_hidden256", HIDDEN_DIM=256, CLASSIFIER_LR=5e-4),
]


def main():
    dish = load_dish_table(DATA_ROOT)
    print("loaded dishes", len(dish), "exists", int(dish["exists"].sum()))
    rows = []
    for cfg in CONFIGS:
        print("\n" + "=" * 60)
        print("CONFIG", cfg.NAME)
        tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL, use_fast=True)
        tr, va, te = split_frames(dish, cfg)
        print(f"sizes train/val/test = {len(tr)}/{len(va)}/{len(te)}")
        train_loader, val_loader, test_loader = make_loaders(
            tr, va, te, tokenizer, cfg.IMAGE_SIZE, cfg.BATCH_SIZE, cfg.USE_AUG, cfg.MAX_LENGTH
        )
        result = train(cfg, train_loader, val_loader, test_loader)
        rows.append(
            {
                "config": result["name"],
                "val_mae": round(result["val_mae"], 2),
                "test_mae": round(result["test_mae"], 2),
                "best_epoch": result["best_epoch"],
                "trainable_params": result["trainable_params"],
                "sec_per_epoch": round(result["sec_per_epoch"], 1),
                "image_size": cfg.IMAGE_SIZE,
                "hidden_dim": cfg.HIDDEN_DIM,
                "use_aug": cfg.USE_AUG,
                "unfreeze_text_last": cfg.UNFREEZE_TEXT_LAST,
                "unfreeze_image_last": cfg.UNFREEZE_IMAGE_LAST,
            }
        )
        # free memory
        del result
        import gc

        gc.collect()

    summary = pd.DataFrame(rows).sort_values("val_mae")
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    best = summary.iloc[0].to_dict()
    (OUT_DIR / "best_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY (sorted by val_mae) ===")
    print(summary.to_string(index=False))
    print("BEST:", best["config"], "val_mae=", best["val_mae"])


if __name__ == "__main__":
    main()
