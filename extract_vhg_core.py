from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import gc
import glob
import json
import os
import re
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
from timm.data import resolve_model_data_config

DEFAULT_MAPPING_CSV = "data/BraTS-Path-2026-Train-Patch-Patient-Slide-Mapping.csv"
DEFAULT_TRAIN_GLOB = "data/train/shard-*.tar"
DEFAULT_VAL_GLOB = "data/val-shard-*.tar"
DEFAULT_ARTIFACTS = "artifacts"

FOUNDATIONS = ("virchow2", "hoptimus1", "genbiopathfm")
SPLITS = ("val", "train")

DEVICE = "auto"
BATCH_SIZE = 128
NUM_WORKERS = 4
TRAIN_UNIT_SHARDS = 1
VAL_UNIT_SHARDS = 1
FLUSH_EVERY_BATCHES = 50
STORAGE_DTYPE = "float16"
USE_AMP = True
L2_NORMALIZE = True
COMPILE_MODEL = False
RESTART = False
RESTART_FINAL = False
FORCE_REEXTRACT = False

FOUNDATION_SPECS = {
    "virchow2": {
        "hf_id": "hf-hub:paige-ai/Virchow2",
        "input_size": 224,
        "embedding_dim": 2560,
    },
    "hoptimus1": {
        "hf_id": "hf-hub:bioptimus/H-optimus-1",
        "input_size": 224,
        "embedding_dim": 1536,
        "mean": (0.707223, 0.578729, 0.703617),
        "std": (0.211883, 0.230117, 0.177517),
    },
    "genbiopathfm": {
        "hf_id": "genbio-ai/genbio-pathfm",
        "input_size": 224,
        "embedding_dim": 4608,
        "mean": (0.697, 0.575, 0.728),
        "std": (0.188, 0.240, 0.187),
    },
}


def auto_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def discover_shards(pattern):
    paths = sorted(Path().glob(pattern))
    if not paths:
        paths = [Path(p) for p in sorted(glob.glob(pattern))]
    return paths


def read_mapping(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    required = {"Name", "Patient", "Slide"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mapping CSV missing columns: {sorted(missing)}")
    out = {}
    for name, patient, slide in zip(df["Name"].astype(str), df["Patient"], df["Slide"]):
        out[name] = (str(int(patient)), str(int(slide)))
    return out


def safe_patient(value):
    if value is None or str(value) in {"", "-1", "nan", "None", "unmapped"}:
        return "patient-unmapped"
    try:
        return f"patient-{int(value):03d}"
    except Exception:
        return f"patient-{re.sub(r'[^A-Za-z0-9_.-]+', '-', str(value))}"


def safe_slide(value):
    if value is None or str(value) in {"", "-1", "nan", "None", "unmapped"}:
        return "slide-unmapped"
    try:
        return f"slide-{int(value):04d}"
    except Exception:
        return f"slide-{re.sub(r'[^A-Za-z0-9_.-]+', '-', str(value))}"


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    np.savez(tmp, **arrays)
    tmp_npz = Path(str(tmp) if str(tmp).endswith(".npz") else str(tmp) + ".npz")
    tmp_npz.replace(path)


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def build_transform(foundation, model):
    from torchvision import transforms

    spec = FOUNDATION_SPECS[foundation]

    if foundation == "genbiopathfm":
        interpolation = transforms.InterpolationMode.BILINEAR
        mean = spec["mean"]
        std = spec["std"]
    elif foundation == "hoptimus1":
        interpolation = transforms.InterpolationMode.BICUBIC
        mean = spec["mean"]
        std = spec["std"]
    else:
        interpolation = transforms.InterpolationMode.BICUBIC
        cfg = resolve_model_data_config(model)
        mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))
        std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))

    return transforms.Compose(
        [
            transforms.Resize(
                (spec["input_size"], spec["input_size"]),
                interpolation=interpolation,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_foundation(foundation, device):
    spec = FOUNDATION_SPECS[foundation]

    if foundation == "hoptimus1":
        model = timm.create_model(
            spec["hf_id"],
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
    elif foundation == "genbiopathfm":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            spec["hf_id"],
            trust_remote_code=True,
        )
    else:
        try:
            model = timm.create_model(
                spec["hf_id"],
                pretrained=True,
                num_classes=0,
            )
        except TypeError:
            model = timm.create_model(
                spec["hf_id"],
                pretrained=True,
            )

    model.eval().to(device)

    if COMPILE_MODEL:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    transform = build_transform(foundation, model)
    return model, transform


def pool_tokens(out):
    if isinstance(out, dict):
        for key in ("x_norm_clstoken", "pooled", "global_pool", "features"):
            value = out.get(key)
            if torch.is_tensor(value):
                return value
        for value in out.values():
            if torch.is_tensor(value):
                out = value
                break

    if isinstance(out, (tuple, list)):
        out = out[0]

    if torch.is_tensor(out) and out.ndim == 3:
        return out[:, 0]

    if torch.is_tensor(out) and out.ndim > 3:
        return out.flatten(1)

    return out


def extract_features(model, images, foundation):
    if foundation == "virchow2":
        out = model.forward_features(images)

        if isinstance(out, dict):
            cls = out.get("x_norm_clstoken")
            patch = out.get("x_norm_patchtokens")

            if torch.is_tensor(cls) and torch.is_tensor(patch):
                emb = torch.cat([cls, patch.mean(dim=1)], dim=1)
            else:
                emb = pool_tokens(out)
        elif torch.is_tensor(out) and out.ndim == 3:
            cls = out[:, 0]
            patch = out[:, 5:] if out.shape[1] > 260 else out[:, 1:]
            emb = torch.cat([cls, patch.mean(dim=1)], dim=1)
        else:
            emb = pool_tokens(out)

    elif foundation == "genbiopathfm":
        emb = model(images)

    else:
        emb = pool_tokens(model(images))

    if not torch.is_tensor(emb):
        raise TypeError(f"{foundation}: model returned {type(emb)}")

    if emb.ndim != 2:
        emb = emb.flatten(1)

    expected_dim = FOUNDATION_SPECS[foundation]["embedding_dim"]

    if emb.shape[1] != expected_dim:
        raise RuntimeError(
            f"{foundation}: expected embedding dim {expected_dim}, got {emb.shape[1]}"
        )

    if L2_NORMALIZE:
        emb = F.normalize(emb.float(), p=2, dim=1)

    return emb


class SampleConverter:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        key = sample.get("__key__", "")
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        key = str(key)

        image = None
        for field in ("jpg", "jpeg", "png", "webp"):
            if field in sample:
                image = sample[field]
                break

        if image is None:
            raise ValueError(f"{key}: no image field")

        if not isinstance(image, Image.Image):
            raise ValueError(f"{key}: expected PIL image, got {type(image)}")

        label = -1
        for field in ("cls", "txt"):
            if field in sample:
                value = sample[field]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                label = int(str(value).strip())
                break

        return key, self.transform(image.convert("RGB")), label


def make_dataset(shards, transform):
    return (
        wds.WebDataset(
            [str(path) for path in shards],
            shardshuffle=False,
            empty_check=False,
        )
        .decode("pil")
        .map(SampleConverter(transform))
    )


def chunk_shards(shards, size):
    size = max(1, int(size))
    return [shards[i : i + size] for i in range(0, len(shards), size)]


def unit_done(unit_dir):
    return unit_dir / "_UNIT_DONE.json"


def block_done(block_dir):
    return block_dir / "_DONE.json"


def completed_block_dirs(unit_dir):
    if not unit_dir.exists():
        return []
    return [
        path for path in sorted(unit_dir.glob("block-*")) if block_done(path).exists()
    ]


def read_completed_keys(unit_dir):
    keys = set()

    for block_dir in completed_block_dirs(unit_dir):
        try:
            meta = json.loads(block_done(block_dir).read_text())
            keys.update(map(str, meta.get("keys", [])))
        except Exception:
            for part in block_dir.glob("patient-*/slide-*.part.npz"):
                try:
                    d = np.load(part, allow_pickle=True)
                    keys.update(map(str, d["names"].tolist()))
                except Exception:
                    pass

    return keys


def next_block_index(unit_dir):
    maximum = -1

    for path in unit_dir.glob("block-*"):
        match = re.match(r"block-(\d+)$", path.name)
        if match:
            maximum = max(maximum, int(match.group(1)))

    return maximum + 1


def write_block(
    unit_dir,
    final_block_dir,
    foundation,
    split,
    unit_name,
    block_index,
    buffers,
    block_keys,
    elapsed_s,
):
    tmp_dir = unit_dir / f".{final_block_dir.name}.tmp.{os.getpid()}"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)

    n_files = 0
    n_patches = 0

    for (patient_dir, slide_stem), buffer in buffers.items():
        if not buffer["names"]:
            continue

        X = np.concatenate(buffer["X"], axis=0)
        y = np.concatenate(buffer["y"], axis=0)
        names = np.asarray(buffer["names"], dtype=object)

        out = tmp_dir / patient_dir / f"{slide_stem}.part.npz"

        atomic_npz(
            out,
            X=X,
            y=y,
            names=names,
            patient=np.asarray(patient_dir, dtype=object),
            slide=np.asarray(slide_stem, dtype=object),
        )

        n_files += 1
        n_patches += int(X.shape[0])

    meta = {
        "foundation": foundation,
        "split": split,
        "unit": unit_name,
        "block": f"block-{block_index:06d}",
        "n_patches": n_patches,
        "n_patient_slide_files": n_files,
        "keys": list(map(str, block_keys)),
        "dtype": STORAGE_DTYPE,
        "elapsed_s_since_unit_start": round(float(elapsed_s), 3),
    }

    atomic_json(tmp_dir / "_DONE.json", meta)

    if final_block_dir.exists():
        shutil.rmtree(final_block_dir)

    tmp_dir.replace(final_block_dir)
    return meta


def check_features(X, context):
    finite = np.isfinite(X.reshape(X.shape[0], -1)).all(axis=1)
    n_bad = int((~finite).sum())

    if n_bad == 0:
        return X

    fraction = n_bad / max(len(X), 1)

    if fraction > 0.10:
        raise RuntimeError(f"{context}: {n_bad}/{len(X)} embeddings are NaN/Inf")

    out = X.copy()
    out[~finite] = 0.0
    return out


def process_unit(
    unit_idx,
    unit_shards,
    split,
    foundation,
    model,
    transform,
    mapping,
    parts_root,
    device,
):
    unit_name = f"unit-{unit_idx:06d}"
    unit_dir = parts_root / unit_name
    marker = unit_done(unit_dir)

    if marker.exists():
        return json.loads(marker.read_text())

    unit_dir.mkdir(parents=True, exist_ok=True)

    for block_dir in sorted(unit_dir.glob("block-*")):
        if not block_done(block_dir).exists():
            shutil.rmtree(block_dir, ignore_errors=True)

    done_keys = read_completed_keys(unit_dir)
    workers = min(max(0, NUM_WORKERS), len(unit_shards))

    if len(unit_shards) == 1 and workers > 1:
        workers = 1

    dataset = make_dataset(unit_shards, transform)

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "shuffle": False,
        "drop_last": False,
        "persistent_workers": workers > 0,
    }

    if workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    loader = DataLoader(dataset, **loader_kwargs)

    amp_enabled = USE_AMP and device.type in {"cuda", "mps"}
    amp_dtype = torch.float16
    np_dtype = np.float16 if STORAGE_DTYPE == "float16" else np.float32
    context = f"{foundation}/{split}/{unit_name}"

    buffers = defaultdict(lambda: {"X": [], "y": [], "names": []})
    block_keys = []
    block_idx = next_block_index(unit_dir)
    block_batch_count = 0
    n_seen = 0
    n_new = 0
    n_skipped = 0
    started = time.time()

    for keys, images, labels in tqdm(loader, desc=context, leave=False):
        keys = [str(key) for key in keys]
        n_seen += len(keys)

        keep = (
            [i for i, key in enumerate(keys) if key not in done_keys]
            if done_keys
            else list(range(len(keys)))
        )

        if not keep:
            n_skipped += len(keys)
            continue

        if len(keep) < len(keys):
            n_skipped += len(keys) - len(keep)
            index = torch.as_tensor(keep, dtype=torch.long)
            images = images.index_select(0, index)
            labels = labels.index_select(0, index)
            keys = [keys[i] for i in keep]

        images = images.to(device, non_blocking=True)

        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                features = extract_features(model, images, foundation)

        X = features.detach().cpu().float().numpy()
        X = check_features(X, context).astype(np_dtype, copy=False)
        y = labels.detach().cpu().numpy().astype(np.int16, copy=False)

        if mapping is None:
            patients = ["unmapped"] * len(keys)
            slides = ["unmapped"] * len(keys)
        else:
            patients = []
            slides = []

            for key in keys:
                patient_slide = mapping.get(key)

                if patient_slide is None:
                    patients.append("unmapped")
                    slides.append("unmapped")
                else:
                    patients.append(patient_slide[0])
                    slides.append(patient_slide[1])

        groups = defaultdict(list)

        for i, (patient, slide) in enumerate(zip(patients, slides)):
            groups[(safe_patient(patient), safe_slide(slide))].append(i)

        for group, indices in groups.items():
            ia = np.asarray(indices, dtype=np.int64)
            buffer = buffers[group]
            buffer["X"].append(X[ia])
            buffer["y"].append(y[ia])
            buffer["names"].extend(keys[i] for i in indices)

        block_keys.extend(keys)
        n_new += len(keys)
        block_batch_count += 1

        if block_batch_count >= FLUSH_EVERY_BATCHES:
            final_block_dir = unit_dir / f"block-{block_idx:06d}"

            write_block(
                unit_dir,
                final_block_dir,
                foundation,
                split,
                unit_name,
                block_idx,
                buffers,
                block_keys,
                time.time() - started,
            )

            done_keys.update(block_keys)
            buffers = defaultdict(lambda: {"X": [], "y": [], "names": []})
            block_keys = []
            block_batch_count = 0
            block_idx += 1
            gc.collect()

    if block_keys:
        final_block_dir = unit_dir / f"block-{block_idx:06d}"

        write_block(
            unit_dir,
            final_block_dir,
            foundation,
            split,
            unit_name,
            block_idx,
            buffers,
            block_keys,
            time.time() - started,
        )

    block_metas = [
        json.loads(block_done(path).read_text())
        for path in completed_block_dirs(unit_dir)
    ]

    n_total = sum(int(meta.get("n_patches", 0)) for meta in block_metas)
    elapsed = time.time() - started

    meta = {
        "unit": unit_name,
        "split": split,
        "foundation": foundation,
        "shards": [str(path) for path in unit_shards],
        "n_shards": len(unit_shards),
        "n_patches": int(n_total),
        "n_seen_this_run": n_seen,
        "n_new_this_run": n_new,
        "n_skipped_this_run": n_skipped,
        "n_blocks": len(block_metas),
        "elapsed_s_this_run": round(elapsed, 3),
        "new_patches_per_s_this_run": round(n_new / max(elapsed, 1e-9), 3),
        "dtype": STORAGE_DTYPE,
        "batch_size": BATCH_SIZE,
        "num_workers": workers,
        "l2_normalize": L2_NORMALIZE,
    }

    atomic_json(marker, meta)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return meta


def collect_part_groups(parts_root):
    groups = defaultdict(list)

    for unit_dir in sorted(parts_root.glob("unit-*")):
        if not unit_done(unit_dir).exists():
            continue

        for block_dir in completed_block_dirs(unit_dir):
            for part in block_dir.glob("patient-*/slide-*.part.npz"):
                groups[
                    (
                        part.parent.name,
                        part.stem.replace(".part", ""),
                    )
                ].append(part)

    return groups


def consolidate(split, foundation, parts_root, final_root):
    manifest_path = final_root / "manifest.json"

    if manifest_path.exists() and not RESTART_FINAL:
        return json.loads(manifest_path.read_text())

    if RESTART_FINAL and final_root.exists():
        shutil.rmtree(final_root)

    final_root.mkdir(parents=True, exist_ok=True)
    groups = collect_part_groups(parts_root)

    if not groups:
        raise RuntimeError(f"No completed parts under {parts_root}")

    n_patches = 0
    n_files = 0

    for (patient_dir, slide_stem), part_paths in tqdm(
        sorted(groups.items()),
        desc=f"consolidate {foundation}/{split}",
    ):
        out_path = final_root / patient_dir / f"{slide_stem}.npz"

        if out_path.exists() and not RESTART_FINAL:
            try:
                n_patches += int(len(np.load(out_path, allow_pickle=True)["names"]))
                n_files += 1
                continue
            except Exception:
                out_path.unlink(missing_ok=True)

        Xs = []
        ys = []
        names = []

        for part in sorted(part_paths):
            data = np.load(part, allow_pickle=True)
            Xs.append(data["X"])
            ys.append(data["y"])
            names.extend(data["names"].tolist())

        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        names = np.asarray(names, dtype=object)

        order = np.argsort(names.astype(str), kind="mergesort")
        X = X[order]
        y = y[order]
        names = names[order]

        if len(names) > 1:
            _, reverse_indices = np.unique(
                names[::-1].astype(str),
                return_index=True,
            )
            keep = np.sort(len(names) - 1 - reverse_indices)
            X = X[keep]
            y = y[keep]
            names = names[keep]

        atomic_npz(
            out_path,
            X=X,
            y=y,
            names=names,
            patient=np.asarray(patient_dir, dtype=object),
            slide=np.asarray(slide_stem, dtype=object),
        )

        n_patches += int(X.shape[0])
        n_files += 1

    meta = {
        "split": split,
        "foundation": foundation,
        "n_patient_slide_files": n_files,
        "n_patches": n_patches,
        "dtype": STORAGE_DTYPE,
        "l2_normalize": L2_NORMALIZE,
    }

    atomic_json(manifest_path, meta)
    return meta


def extract_split(foundation, split, shards, mapping, device):
    artifacts = Path(DEFAULT_ARTIFACTS)
    parts_root = artifacts / "embedding_parts" / foundation / split
    final_root = artifacts / "embeddings_by_patient_slide" / foundation / split

    if RESTART:
        if parts_root.exists():
            shutil.rmtree(parts_root)
        if final_root.exists():
            shutil.rmtree(final_root)

    if (final_root / "manifest.json").exists() and not RESTART and not FORCE_REEXTRACT:
        print(f"[skip] {foundation}/{split}")
        return

    unit_size = VAL_UNIT_SHARDS if split == "val" else TRAIN_UNIT_SHARDS
    units = chunk_shards(shards, unit_size)

    print(f"[extract] {foundation}/{split} shards={len(shards)} units={len(units)}")

    model, transform = load_foundation(foundation, device)

    for unit_idx, unit in enumerate(units):
        meta = process_unit(
            unit_idx,
            unit,
            split,
            foundation,
            model,
            transform,
            mapping,
            parts_root,
            device,
        )

        print(
            f"[done] {foundation}/{split}/{meta['unit']} patches={meta['n_patches']:,}"
        )

    meta = consolidate(
        split,
        foundation,
        parts_root,
        final_root,
    )

    print(
        f"[consolidated] {foundation}/{split} "
        f"patches={meta['n_patches']:,} "
        f"files={meta['n_patient_slide_files']:,}"
    )

    del model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    device = auto_device() if DEVICE == "auto" else torch.device(DEVICE)

    train_shards = discover_shards(DEFAULT_TRAIN_GLOB)
    val_shards = discover_shards(DEFAULT_VAL_GLOB)

    if not train_shards:
        raise FileNotFoundError(DEFAULT_TRAIN_GLOB)

    if not val_shards:
        raise FileNotFoundError(DEFAULT_VAL_GLOB)

    mapping = read_mapping(DEFAULT_MAPPING_CSV)

    print(f"device={device}")
    print(f"train_shards={len(train_shards)}")
    print(f"val_shards={len(val_shards)}")
    print(f"mapping_rows={len(mapping):,}")

    for foundation in FOUNDATIONS:
        if "val" in SPLITS:
            extract_split(
                foundation,
                "val",
                val_shards,
                None,
                device,
            )

        if "train" in SPLITS:
            extract_split(
                foundation,
                "train",
                train_shards,
                mapping,
                device,
            )


if __name__ == "__main__":
    main()
