"""Поиск / скачивание / безопасная распаковка датасета в data/."""

from __future__ import annotations

import json
import os
import re
import shutil
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

DATASET_DIRNAME = "data"
LEGACY_DIRNAME = "nutrition"
DATASET_ZIPNAME = "nutrition.zip"
YANDEX_DISK_URL = "https://disk.yandex.ru/d/kz9g5msVqtahiw"
MAX_ZIP_UNCOMPRESSED = 5 * 1024**3
ALLOWED_PUBLIC_PREFIXES = (
    "https://disk.yandex.ru/",
    "https://disk.yandex.com/",
    "https://yadi.sk/",
)
ALLOWED_DOWNLOAD_HOST_SUFFIXES = (
    ".yandex.net",
    ".yandex.ru",
    ".yadi.sk",
    "yandex.net",
    "yandex.ru",
    "yadi.sk",
)


def is_data_root(p: Path) -> bool:
    return (
        p.is_dir()
        and (p / "dish.csv").is_file()
        and (p / "ingredients.csv").is_file()
        and (p / "images").is_dir()
    )


def notebook_dir(cwd: Path | None = None) -> Path:
    cwd = (cwd or Path.cwd()).resolve()
    if (cwd / "avo_m25_555_nutrition.ipynb").is_file() or is_data_root(cwd / DATASET_DIRNAME):
        return cwd
    for base in [cwd, *cwd.parents]:
        if (base / "avo_m25_555_nutrition.ipynb").is_file():
            return base
        nested = base / "avo_Nutrition_rep"
        if (nested / "avo_m25_555_nutrition.ipynb").is_file():
            return nested
    return cwd


def promote_legacy(folder: Path) -> None:
    legacy = folder / LEGACY_DIRNAME
    dest = folder / DATASET_DIRNAME
    if is_data_root(legacy) and not dest.exists():
        legacy.rename(dest)
        print("переименовано", legacy, "→", dest)


def _host_allowed(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if not host:
        return False
    return any(host == s.lstrip(".") or host.endswith(s if s.startswith(".") else f".{s}") for s in ALLOWED_DOWNLOAD_HOST_SUFFIXES)


def _public_url_ok(url: str) -> bool:
    u = url.strip()
    return u.startswith(ALLOWED_PUBLIC_PREFIXES) and _host_allowed(u)


class _HostGuardRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _host_allowed(newurl):
            raise urllib.error.HTTPError(newurl, code, "redirect host not allowed", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    https = urllib.request.HTTPSHandler(context=ctx)
    return urllib.request.build_opener(https, _HostGuardRedirect)


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    return (info.external_attr >> 16) & 0o170000 == 0o120000


def _safe_extractall(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    total = 0
    for info in zf.infolist():
        if _is_zip_symlink(info):
            raise ValueError(f"Zip: symlink запрещён {info.filename!r}")
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
            raise ValueError(f"Zip Slip: абсолютный путь {info.filename!r}")
        target = (dest / name).resolve()
        try:
            target.relative_to(dest)
        except ValueError as exc:
            raise ValueError(f"Zip Slip: {info.filename!r} → {target}") from exc
        total += max(0, info.file_size)
        if total > MAX_ZIP_UNCOMPRESSED:
            raise ValueError("Zip слишком большой после распаковки (zip bomb)")
    zf.extractall(dest)


def _allowed_data_dirs(nb_dir: Path) -> set[Path]:
    return {
        (nb_dir / DATASET_DIRNAME).resolve(),
        (nb_dir / "avo_Nutrition_rep" / DATASET_DIRNAME).resolve(),
        (Path("/content") / DATASET_DIRNAME).resolve(),
        (Path("/content/avo_Nutrition_rep") / DATASET_DIRNAME).resolve(),
    }


def _guard_dest(dest: Path, nb_dir: Path) -> Path:
    dest = dest.resolve()
    if dest not in _allowed_data_dirs(nb_dir):
        raise ValueError(f"Распаковка только в data/: {dest}")
    return dest


def _flatten_to_dest(dest: Path, found: Path) -> Path:
    dest = dest.resolve()
    found = found.resolve()
    if found == dest:
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    if found.parent == dest:
        for item in found.iterdir():
            target = dest / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
        try:
            found.rmdir()
        except OSError:
            pass
        if is_data_root(dest):
            return dest
    return found


def extract_zip(zpath: Path, dest: Path, nb_dir: Path) -> Path | None:
    if not zpath.is_file():
        return None
    dest = _guard_dest(dest, nb_dir)
    with zpath.open("rb") as f:
        if f.read(2) != b"PK":
            print("Файл не ZIP:", zpath)
            return None
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        _safe_extractall(zf, dest)
    if is_data_root(dest):
        return dest
    for child in dest.iterdir():
        if child.is_dir() and is_data_root(child):
            return _flatten_to_dest(dest, child)
    for dish in dest.glob("*/dish.csv"):
        if is_data_root(dish.parent):
            return _flatten_to_dest(dest, dish.parent)
    return None


def download_yandex_disk_zip(public_url: str, dest_zip: Path) -> Path | None:
    if not public_url.strip():
        return None
    if not _public_url_ok(public_url):
        print("Отказ: публичная ссылка не Яндекс.Диск HTTPS:", public_url)
        return None
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    api = (
        "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key="
        + urllib.parse.quote(public_url, safe="")
    )
    print("Запрашиваю прямую ссылку Яндекс.Диска...")
    opener = _opener()
    try:
        with opener.open(api, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        href = payload.get("href")
        if not href or not str(href).startswith("https://") or not _host_allowed(href):
            print("Отказ: href не HTTPS Яндекса:", href)
            return None

        def _progress(block_num: int, block_size: int, total_size: int) -> None:
            if total_size <= 0:
                return
            done = block_num * block_size
            pct = min(100.0, 100.0 * done / total_size)
            if block_num % 200 == 0 or done >= total_size:
                print(f"скачивание nutrition.zip: {pct:5.1f}%", flush=True)

        print("Скачиваю", dest_zip)
        tmp = dest_zip.with_suffix(".part")
        with opener.open(href, timeout=600) as resp, tmp.open("wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
                got += len(chunk)
                if total:
                    _progress(got, 1, total)
        if tmp.is_file() and tmp.stat().st_size > 1024:
            tmp.replace(dest_zip)
            print("Скачано:", dest_zip, dest_zip.stat().st_size, "байт")
            return dest_zip
        print("Скачанный файл пустой или слишком маленький.")
        if tmp.exists():
            tmp.unlink()
        return None
    except Exception as e:
        print("Не удалось скачать с Яндекс.Диска:", e)
        return None


def pick_local_folder() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        print("Диалог папки недоступен:", e)
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    chosen = filedialog.askdirectory(title="Выберите папку data (dish.csv + ingredients.csv + images/)")
    root.destroy()
    return Path(chosen) if chosen else None


def upload_zip_colab(save_as: Path) -> Path | None:
    from google.colab import files  # type: ignore

    print("Загрузите nutrition.zip (Choose Files)…")
    uploaded = files.upload()
    if not uploaded:
        return None
    name, raw = next(iter(uploaded.items()))
    save_as.parent.mkdir(parents=True, exist_ok=True)
    target = save_as if str(save_as).lower().endswith(".zip") else save_as / DATASET_ZIPNAME
    Path(target).write_bytes(raw)
    print("Сохранено:", target, "из", name)
    return Path(target)


def _find_data_near(nb_dir: Path) -> Path | None:
    for c in (
        nb_dir / DATASET_DIRNAME,
        nb_dir / "avo_Nutrition_rep" / DATASET_DIRNAME,
        Path("/content") / DATASET_DIRNAME,
        Path("/content/avo_Nutrition_rep") / DATASET_DIRNAME,
        nb_dir / LEGACY_DIRNAME,
        nb_dir / "avo_Nutrition_rep" / LEGACY_DIRNAME,
        Path("/content") / LEGACY_DIRNAME,
    ):
        if is_data_root(c):
            return c
    return None


def resolve_data_root() -> tuple[Path, Path]:
    """Вернуть (NOTEBOOK_DIR, DATA_ROOT)."""
    in_colab = "google.colab" in sys.modules
    nb_dir = notebook_dir()
    print("NOTEBOOK_DIR =", nb_dir)
    promote_legacy(nb_dir)
    promote_legacy(nb_dir / "avo_Nutrition_rep")
    if Path("/content").is_dir():
        promote_legacy(Path("/content"))
        promote_legacy(Path("/content/avo_Nutrition_rep"))

    data_root = _find_data_near(nb_dir)

    zip_candidates = (
        nb_dir / DATASET_ZIPNAME,
        nb_dir / "avo_Nutrition_rep" / DATASET_ZIPNAME,
        Path("/content") / DATASET_ZIPNAME,
    )
    if data_root is None:
        for z in zip_candidates:
            if z.is_file():
                print("Найден zip:", z)
                dest = (Path("/content") if in_colab else nb_dir) / DATASET_DIRNAME
                data_root = extract_zip(z, dest, nb_dir)
                if data_root is not None:
                    break

    if data_root is None and YANDEX_DISK_URL.strip():
        zip_path = (Path("/content") if in_colab else nb_dir) / DATASET_ZIPNAME
        print("Локально data/ не найдена — скачиваю nutrition.zip с Яндекс.Диска.")
        got = download_yandex_disk_zip(YANDEX_DISK_URL, zip_path)
        if got is not None:
            dest = (Path("/content") if in_colab else nb_dir) / DATASET_DIRNAME
            data_root = extract_zip(got, dest, nb_dir)

    if data_root is None:
        print("data/ не найдена — выберите папку или загрузите nutrition.zip.")
        if in_colab:
            z = upload_zip_colab(Path("/content") / DATASET_ZIPNAME)
            if z is not None:
                data_root = extract_zip(z, Path("/content") / DATASET_DIRNAME, nb_dir)
        else:
            picked = pick_local_folder()
            if picked is not None:
                print("Выбрано:", picked)
                if is_data_root(picked):
                    data_root = picked
                elif is_data_root(picked / DATASET_DIRNAME):
                    data_root = picked / DATASET_DIRNAME
                elif is_data_root(picked / LEGACY_DIRNAME):
                    data_root = picked / LEGACY_DIRNAME
                else:
                    for z in list(picked.glob(DATASET_ZIPNAME)) + list(picked.glob("*.zip")):
                        data_root = extract_zip(z, nb_dir / DATASET_DIRNAME, nb_dir)
                        if data_root is not None:
                            break

    if data_root is None or not is_data_root(data_root):
        raise FileNotFoundError(
            "Датасет не найден. Нужна папка data/ (dish.csv, ingredients.csv, images/), "
            "nutrition.zip, Яндекс.Диск, выбор папки или Upload."
        )
    return nb_dir, data_root.resolve()
