from __future__ import annotations

import gc
import hashlib
import json
import shutil
import sqlite3
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


SEED = 42
N_CLASSES = 10
CLASS_IDS = np.arange(N_CLASSES, dtype=np.int64)
NOTA_INDEX = 9

GROUP_BY = "patient"
N_SPLITS = 5
FORCE_TRAIN_PATIENTS: Tuple[str, ...] = (
    "patient-059",  # NOTA-only patient
    "patient-072",
)

# Foundations actually needed by the configs wired in below.
FOUNDATIONS: List[str] = [
    "virchow2",
    "hoptimus1",
    "genbiopathfm",
    "provgigapath",
    "conch",
    "neurofm",
    "h0mini",
    "phikonv2",
    "uni2",
]

RANK_COLUMNS = [
    "f1_per_class_average",
    "mcc",
    "recall_per_class_average",
    "specificity_per_class_weighted",
    "auroc_per_class_weighted",
    "accuracy_global",
]
METRIC_NAMES = tuple(RANK_COLUMNS)
METRIC_SUMMARY_COLUMNS = tuple(
    f"{metric}_{stat}" for metric in METRIC_NAMES for stat in ("mean", "std")
)


@dataclass(frozen=True)
class SweepSettings:
    artifacts: Path = Path("artifacts")
    embedding_root: Path = Path("artifacts/embeddings_by_patient_slide")
    out_dir: Path = Path("X_SWEEP_OOF")
    split: str = "train"
    partial_suffix: str = "aug_partial"
    max_files: int | None = None
    overwrite: bool = False

    # Special train-only augmentation token. Unlike stainaug/stainaug_local,
    # Ivy GAP is not materialized into the CV patch universe because it has no
    # BraTS patient/slide grouping. Whenever an active config includes "ivy"
    # in aug_artifact_suffixes, these roots are loaded once and appended only to
    # each fold's training fit. Validation folds remain BraTS-only.
    ivy_virchow2_root: Path = Path(
        "/Users/agata/competitions/brats/ivygap_embeddings_raw/virchow2"
    )
    ivy_hoptimus1_root: Path = Path(
        "/Users/agata/competitions/brats/ivygap_embeddings_raw/hoptimus1"
    )
    ivy_max_per_class: int = 8120
    ivy_seed: int = SEED
    ivy_norm_mode: str = "auto"


SWEEP = SweepSettings()


@dataclass(frozen=True)
class ProbeConfig:
    name: str
    estimator: Any
    transform: str = "l2"
    only_foundations: Tuple[str, ...] = ()
    # Empty tuple means baseline/no augmentation. Each regular entry is the
    # suffix in <foundation>_<suffix>, e.g. "stainaug" or "stainaug_local".
    # Multiple regular entries are concatenated into one temporary merged
    # foundation. The special token "ivy" is train-only: Ivy rows are appended
    # to each fold's training fit and are never included in validation OOF rows.
    aug_artifact_suffixes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CompositeProbeConfig:
    name: str
    foundation_names: Tuple[str, ...]
    estimator: Any
    transform: str = "none"
    # Applied to every foundation in foundation_names unless overridden below.
    aug_artifact_suffixes: Tuple[str, ...] = ()
    # Optional per-foundation override, e.g.
    # {"hoptimus1": ("stainaug",), "virchow2": ()}.
    per_foundation_aug_artifact_suffixes: Mapping[str, Tuple[str, ...]] = field(
        default_factory=dict
    )


# -----------------------------------------------------------------------------
# Config identity (independent namespace from the search sweep -- different
# protocol, different registry, so ids must not collide with it)
# -----------------------------------------------------------------------------


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(k): _canonicalize(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, np.ndarray):
        return _canonicalize(value.tolist())
    if hasattr(value, "get_params"):
        try:
            return {
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "params": _canonicalize(value.get_params(deep=False)),
            }
        except Exception:
            pass
    return {
        "class": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": str(value),
    }


def embedding_manifest_fingerprint(
    root: Path, foundation_names: Tuple[str, ...]
) -> str:
    h = hashlib.sha1()
    for foundation in foundation_names:
        manifest = root / foundation / SWEEP.split / "manifest.json"
        h.update(foundation.encode("utf-8"))
        h.update(b"\0")
        h.update(manifest.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def config_name(foundation_label: str, probe: str, transform: str) -> str:
    return f"{foundation_label}/{probe}/{transform}"


def config_id(
    name: str,
    estimator: Any,
    foundation_names: Tuple[str, ...],
    manifest_fingerprint: str,
    train_augmentation_fingerprint: str = "",
) -> str:
    spec = {
        "schema": "oof-sweep-v1",
        "config_name": name,
        "foundation_names": foundation_names,
        "embedding_manifest": manifest_fingerprint,
        "train_augmentation_fingerprint": train_augmentation_fingerprint,
        "estimator": _canonicalize(estimator),
        "transform": name.rsplit("/", 1)[-1],
        "cv": {
            "group_by": GROUP_BY,
            "n_splits": N_SPLITS,
            "seed": SEED,
            "force_train_patients": FORCE_TRAIN_PATIENTS,
            "max_files": SWEEP.max_files,
            "n_classes": N_CLASSES,
        },
    }
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def composite_foundation_label(foundation_names: Tuple[str, ...]) -> str:
    return "composite__" + "+".join(foundation_names)


# -----------------------------------------------------------------------------
# Data loading (identical protocol to the search sweep)
# -----------------------------------------------------------------------------


def parse_patient_slide(path: Path) -> Tuple[str, str]:
    return path.parent.name, path.stem


def discover_available_foundations(root: Path) -> List[str]:
    if not root.exists():
        return []
    return [
        p.name
        for p in sorted(root.iterdir())
        if (p / "train" / "manifest.json").exists()
    ]


def load_embeddings(
    root: Path, foundation: str, split: str = "train", max_files: int | None = None
):
    base = root / foundation / split
    manifest = base / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    files = sorted(base.glob("patient-*/slide-*.npz"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No slide files found in {base}")

    Xs, ys, names, patients, slides, slide_groups = [], [], [], [], [], []
    for fp in tqdm(files, desc=f"load {foundation}/{split}"):
        d = np.load(fp, allow_pickle=True)
        X = d["X"]
        y = d["y"].astype(np.int16)
        nms = d["names"].astype(str)
        patient, slide = parse_patient_slide(fp)
        Xs.append(X)
        ys.append(y)
        names.append(nms)
        patients.extend([patient] * len(nms))
        slides.extend([slide] * len(nms))
        slide_groups.extend([f"{patient}:{slide}"] * len(nms))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    names = np.concatenate(names, axis=0).astype(str)
    patients = np.asarray(patients, dtype=object)
    slides = np.asarray(slides, dtype=object)
    slide_groups = np.asarray(slide_groups, dtype=object)

    mask = y >= 0
    X = X[mask].astype(np.float32, copy=False)
    y = y[mask]
    names = names[mask]
    patients = patients[mask]
    slides = slides[mask]
    slide_groups = slide_groups[mask]

    nan_mask = ~np.isfinite(X).all(axis=1)
    if nan_mask.any():
        n_bad = int(nan_mask.sum())
        pct = 100.0 * n_bad / max(len(X), 1)
        print(
            f"[warn] {n_bad:,} patches ({pct:.3f}%) contain NaN/Inf embeddings -- zeroing out."
        )
        X[nan_mask] = 0.0

    return X, y, names, patients, slides, slide_groups


def build_folds(
    y: np.ndarray,
    patients: np.ndarray,
    slide_groups: np.ndarray,
    group_by: str,
    n_splits: int,
    seed: int,
    force_train_patients: Tuple[str, ...] = (),
) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups_all = patients if group_by == "patient" else slide_groups

    forced_mask = (
        np.isin(patients, list(force_train_patients))
        if force_train_patients
        else np.zeros(len(y), dtype=bool)
    )
    forced_idx = np.flatnonzero(forced_mask)
    eligible_idx = np.flatnonzero(~forced_mask)
    if len(eligible_idx) == 0:
        raise ValueError(
            "All rows were forced into train; nothing left to build CV folds from."
        )

    groups_eligible = groups_all[eligible_idx]
    y_eligible = y[eligible_idx]

    n_groups = len(np.unique(groups_eligible))
    n_splits = min(int(n_splits), int(n_groups))
    if n_splits < 2:
        raise ValueError(
            f"Need at least two groups for CV; found {n_groups} eligible groups."
        )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for tr_e, va_e in splitter.split(
        np.zeros(len(y_eligible)), y_eligible, groups_eligible
    ):
        tr = eligible_idx[tr_e]
        va = eligible_idx[va_e]
        if len(forced_idx):
            tr = np.concatenate([tr, forced_idx])
        folds.append((tr, va))
    return folds


def align_foundations(
    loaded: Dict[
        str,
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ],
    foundation_order: Tuple[str, ...],
) -> Tuple[
    Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    first = foundation_order[0]
    ref_X, ref_y, ref_names, ref_patients, ref_slides, ref_slide_groups = loaded[first]

    def same_meta(other: Tuple[np.ndarray, ...]) -> bool:
        _, y2, n2, p2 = other[0], other[1], other[2], other[3]
        return (
            len(y2) == len(ref_y)
            and np.array_equal(ref_y, y2)
            and np.array_equal(ref_names, n2)
            and np.array_equal(ref_patients, p2)
        )

    if all(same_meta(loaded[fn]) for fn in foundation_order[1:]):
        print(
            f"  [align] {foundation_order}: identical patch order across foundations -- fast path."
        )
        X_bundle = {fn: loaded[fn][0] for fn in foundation_order}
        return X_bundle, ref_y, ref_names, ref_patients, ref_slides, ref_slide_groups

    print(
        f"  [align] {foundation_order}: patch order differs; aligning by (patient, name) intersection."
    )
    ref_keys = list(zip(ref_patients.tolist(), ref_names.tolist()))
    member_maps: Dict[str, Dict[Tuple[str, str], int]] = {}
    for fn in foundation_order[1:]:
        _, _, fn_names, fn_patients, _, _ = loaded[fn]
        member_maps[fn] = {
            (p, n): i
            for i, (p, n) in enumerate(zip(fn_patients.tolist(), fn_names.tolist()))
        }

    keep_idx: list = []
    aligned_idx: Dict[str, list] = {fn: [] for fn in foundation_order[1:]}
    for i, k in enumerate(ref_keys):
        if all(k in member_maps[fn] for fn in foundation_order[1:]):
            keep_idx.append(i)
            for fn in foundation_order[1:]:
                aligned_idx[fn].append(member_maps[fn][k])

    if not keep_idx:
        raise RuntimeError(
            f"Composite {foundation_order}: no overlapping (patient, name) keys found."
        )

    keep_idx_arr = np.asarray(keep_idx, dtype=np.int64)
    X_bundle: Dict[str, np.ndarray] = {first: ref_X[keep_idx_arr]}
    for fn in foundation_order[1:]:
        idx = np.asarray(aligned_idx[fn], dtype=np.int64)
        fX, fy, _, _, _, _ = loaded[fn]
        X_bundle[fn] = fX[idx]
        if not np.array_equal(ref_y[keep_idx_arr], fy[idx]):
            raise RuntimeError(
                f"Composite: labels disagree after alignment for {fn!r} vs {first!r}."
            )

    return (
        X_bundle,
        ref_y[keep_idx_arr],
        ref_names[keep_idx_arr],
        ref_patients[keep_idx_arr],
        ref_slides[keep_idx_arr],
        ref_slide_groups[keep_idx_arr],
    )


def load_composite_embeddings(
    root: Path,
    foundation_names: Tuple[str, ...],
    max_files: int | None = None,
    output_names: Tuple[str, ...] | None = None,
):
    loaded: Dict[str, Tuple[np.ndarray, ...]] = {}
    for fn in foundation_names:
        loaded[fn] = load_embeddings(root, fn, SWEEP.split, max_files)
        Xf, yf = loaded[fn][0], loaded[fn][1]
        print(
            f"  {fn}: X={Xf.shape}  labels={np.bincount(yf, minlength=N_CLASSES).tolist()}"
        )
    X_bundle, y, names, patients, slides, slide_groups = align_foundations(
        loaded, foundation_names
    )
    if output_names is not None:
        if len(output_names) != len(foundation_names):
            raise ValueError(
                "output_names must have the same length as foundation_names"
            )
        X_bundle = {
            alias: X_bundle[fn] for fn, alias in zip(foundation_names, output_names)
        }
    return X_bundle, y, names, patients, slides, slide_groups


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------


def apply_transform(X_train: np.ndarray, X_val: np.ndarray, transform: str):
    X_train = X_train.astype(np.float32, copy=False)
    X_val = X_val.astype(np.float32, copy=False)
    if transform == "none":
        return X_train, X_val
    if transform == "l2":
        norm = Normalizer(norm="l2")
        return norm.transform(X_train), norm.transform(X_val)
    if transform == "standard":
        sc = StandardScaler(with_mean=True, with_std=True)
        return sc.fit_transform(X_train), sc.transform(X_val)
    raise ValueError(f"Unknown transform {transform!r}")


def slice_bundle(X, idx: np.ndarray):
    if isinstance(X, dict):
        return {k: v[idx] for k, v in X.items()}
    return X[idx]


def apply_transform_bundle(X_train, X_val, transform: str):
    if isinstance(X_train, dict):
        out_tr: Dict[str, np.ndarray] = {}
        out_va: Dict[str, np.ndarray] = {}
        for k in X_train:
            out_tr[k], out_va[k] = apply_transform(X_train[k], X_val[k], transform)
        return out_tr, out_va
    return apply_transform(X_train, X_val, transform)


def _apply_transform_three_arrays(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_extra: np.ndarray | None,
    transform: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_extra_f = None if X_extra is None else np.asarray(X_extra, dtype=np.float32)
    if transform == "none":
        return X_train, X_val, X_extra_f
    if transform == "l2":
        norm = Normalizer(norm="l2")
        out_tr = norm.transform(X_train)
        out_va = norm.transform(X_val)
        out_ex = None if X_extra_f is None else norm.transform(X_extra_f)
        return out_tr, out_va, out_ex
    if transform == "standard":
        # Fit the scaler on BraTS training rows only. Ivy is train-only evidence,
        # not part of the OOF calibration/validation universe.
        sc = StandardScaler(with_mean=True, with_std=True)
        out_tr = sc.fit_transform(X_train)
        out_va = sc.transform(X_val)
        out_ex = None if X_extra_f is None else sc.transform(X_extra_f)
        return out_tr, out_va, out_ex
    raise ValueError(f"Unknown transform {transform!r}")


def transform_fold_with_extras(
    X_train,
    X_val,
    transform: str,
    source_train_extras: Mapping[str, Tuple[np.ndarray, np.ndarray]]
    | Tuple[np.ndarray, np.ndarray]
    | None = None,
):
    """Apply the fold transform to train/val and any train-only extras.

    Extras are transformed with the same per-fold transform fitted on BraTS
    training rows. They are returned separately so estimators that support
    source_train_extras can keep BraTS calibration clean.
    """
    if not source_train_extras:
        Xtr, Xva = apply_transform_bundle(X_train, X_val, transform)
        return Xtr, Xva, None

    if isinstance(X_train, dict):
        out_tr: Dict[str, np.ndarray] = {}
        out_va: Dict[str, np.ndarray] = {}
        out_ex: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        extras = dict(source_train_extras)  # type: ignore[arg-type]
        unknown = set(extras) - set(X_train)
        if unknown:
            raise KeyError(
                f"Train extras supplied for absent source(s): {sorted(unknown)}"
            )
        for k in X_train:
            X_extra = extras[k][0] if k in extras else None
            out_tr[k], out_va[k], extra_t = _apply_transform_three_arrays(
                X_train[k], X_val[k], X_extra, transform
            )
            if k in extras:
                assert extra_t is not None
                out_ex[k] = (
                    extra_t.astype(np.float32, copy=False),
                    np.asarray(extras[k][1], dtype=np.int64),
                )
        return out_tr, out_va, out_ex

    if isinstance(source_train_extras, Mapping):
        if len(source_train_extras) != 1:
            raise ValueError(
                "Array-valued training needs exactly one source extra block."
            )
        X_extra, y_extra = next(iter(source_train_extras.values()))
    else:
        X_extra, y_extra = source_train_extras
    Xtr, Xva, extra_t = _apply_transform_three_arrays(
        X_train, X_val, X_extra, transform
    )
    assert extra_t is not None
    return (
        Xtr,
        Xva,
        (extra_t.astype(np.float32, copy=False), np.asarray(y_extra, dtype=np.int64)),
    )


def _fit_accepts_source_train_extras(model: Any) -> bool:
    try:
        import inspect

        return "source_train_extras" in inspect.signature(model.fit).parameters
    except Exception:
        return False


def append_train_extras_for_plain_estimator(X_train, y_train: np.ndarray, extras):
    """Fallback for ordinary estimators: append train-only extras to X/y.

    This is safe for single-array estimators. For dict-valued estimators without
    source_train_extras support, source-specific extras would need aligned rows;
    we reject that case instead of silently creating a bad composite.
    """
    if not extras:
        return X_train, y_train
    y_train = np.asarray(y_train, dtype=np.int64)
    if isinstance(X_train, dict):
        raise TypeError(
            "This dict-valued estimator does not accept source_train_extras; "
            "train-only Ivy extras would be source-specific and unaligned."
        )
    X_extra, y_extra = extras
    return (
        np.concatenate(
            [
                np.asarray(X_train, dtype=np.float32),
                np.asarray(X_extra, dtype=np.float32),
            ],
            axis=0,
        ),
        np.concatenate([y_train, np.asarray(y_extra, dtype=np.int64)], axis=0),
    )


# -----------------------------------------------------------------------------
# Prediction probabilities and metrics
# -----------------------------------------------------------------------------


def softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64, copy=False)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)


def predict_proba_n(model, X, n_classes: int, class_ids: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        classes = getattr(model, "classes_", class_ids)
        out = np.zeros((p.shape[0], n_classes), dtype=np.float32)
        for j, c in enumerate(classes):
            ci = int(c)
            if 0 <= ci < n_classes:
                out[:, ci] = p[:, j]
        row_sum = out.sum(axis=1, keepdims=True)
        bad = ~np.isfinite(row_sum[:, 0]) | (row_sum[:, 0] <= 0)
        out[bad] = 1.0 / n_classes
        out[~bad] /= row_sum[~bad]
        return out
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)
        if d.ndim == 1:
            d2 = np.zeros((d.shape[0], 2), dtype=np.float32)
            d2[:, 1] = d
            d = d2
        classes = getattr(model, "classes_", class_ids[: d.shape[1]])
        out_scores = np.full((d.shape[0], n_classes), -20.0, dtype=np.float32)
        for j, c in enumerate(classes):
            ci = int(c)
            if 0 <= ci < n_classes and j < d.shape[1]:
                out_scores[:, ci] = d[:, j]
        return softmax(out_scores)
    pred = model.predict(X).astype(int)
    out = np.full((pred.shape[0], n_classes), 1e-6, dtype=np.float32)
    out[np.arange(len(pred)), pred] = 1.0
    out /= out.sum(axis=1, keepdims=True)
    return out


def weighted_specificity(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> float:
    support = np.bincount(y_true.astype(int), minlength=n_classes).astype(np.float64)
    total = support.sum()
    if total <= 0:
        return float("nan")
    vals = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        yt = y_true == c
        yp = y_pred == c
        tn = np.sum((~yt) & (~yp))
        fp = np.sum((~yt) & yp)
        vals[c] = tn / max(tn + fp, 1)
    return float(np.sum(vals * support) / total)


def weighted_multiclass_auc(
    y_true: np.ndarray, proba: np.ndarray, n_classes: int
) -> float:
    labels = np.arange(n_classes)
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(
            roc_auc_score(
                y_true, proba, labels=labels, multi_class="ovr", average="weighted"
            )
        )
    except Exception:
        support = np.bincount(y_true.astype(int), minlength=n_classes).astype(
            np.float64
        )
        aucs, weights = [], []
        for c in range(n_classes):
            yy = (y_true == c).astype(int)
            if yy.min() == yy.max():
                continue
            try:
                aucs.append(float(roc_auc_score(yy, proba[:, c])))
                weights.append(support[c])
            except Exception:
                pass
        if not aucs:
            return float("nan")
        return float(np.sum(np.asarray(aucs) * np.asarray(weights)) / np.sum(weights))


def compute_metrics(
    y_true: np.ndarray, proba: np.ndarray, n_classes: int
) -> Dict[str, float]:
    pred = proba.argmax(axis=1)
    return {
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "recall_per_class_average": float(
            recall_score(y_true, pred, average="macro", zero_division=0)
        ),
        "f1_per_class_average": float(
            f1_score(y_true, pred, average="macro", zero_division=0)
        ),
        "specificity_per_class_weighted": weighted_specificity(y_true, pred, n_classes),
        "auroc_per_class_weighted": weighted_multiclass_auc(y_true, proba, n_classes),
        "accuracy_global": float(accuracy_score(y_true, pred)),
    }


def per_class_f1(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-class F1, in class-index order. NOTA (forced train-only) will
    always read 0.0 here since it never appears in a validation fold --
    that's expected, not a bug; ignore that column when interpreting."""
    pred = proba.argmax(axis=1)
    return f1_score(
        y_true, pred, average=None, zero_division=0, labels=np.arange(n_classes)
    ).astype(np.float64)


# -----------------------------------------------------------------------------
# Probe implementations -- only the estimators used by the configs below.
# Ported as-is from the search sweep.
# -----------------------------------------------------------------------------


def _sgd(
    alpha: float, loss: str = "log_loss", max_iter: int = 20, seed: int = 42
) -> SGDClassifier:
    return SGDClassifier(
        loss=loss,
        alpha=alpha,
        penalty="l2",
        max_iter=max_iter,
        tol=1e-3,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        early_stopping=True,
        validation_fraction=0.05,
        n_iter_no_change=3,
    )


def _sgd_avg_full(
    alpha: float, loss: str = "log_loss", seed: int = 42, max_iter: int = 20
) -> SGDClassifier:
    return SGDClassifier(
        loss=loss,
        penalty="l2",
        alpha=alpha,
        max_iter=max_iter,
        tol=None,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        shuffle=True,
        average=True,
        early_stopping=False,
    )


class LinearProbe(BaseEstimator, ClassifierMixin):
    """L2-normed embedding(s) -> balanced averaged log-loss SGD -> per-class
    thresholds for macro-F1. Handles a single foundation (plain array) AND a
    concat of several ({fm: array} dict)."""

    def __init__(
        self, alpha=3e-5, max_iter=25, grid=15, passes=2, cal_frac=0.10, seed=42
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.grid = grid
        self.passes = passes
        self.cal_frac = cal_frac
        self.seed = seed

    def _stack(self, X):
        def l2(a):
            a = np.asarray(a, np.float32)
            return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)

        if isinstance(X, dict):
            return np.concatenate([l2(X[k]) for k in sorted(X)], axis=1)
        return l2(X)

    def fit(self, X, y):
        y = np.asarray(y, np.int64)
        Z = self._stack(X)
        self.classes_ = CLASS_IDS.copy()
        fit_i, cal_i = next(
            StratifiedShuffleSplit(
                n_splits=1, test_size=self.cal_frac, random_state=self.seed
            ).split(Z, y)
        )
        self.clf_ = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=None,
            class_weight="balanced",
            average=True,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.clf_.fit(Z[fit_i], y[fit_i])
        self.thr_ = self._tune(
            y[cal_i], predict_proba_n(self.clf_, Z[cal_i], N_CLASSES, CLASS_IDS)
        )
        return self

    def _tune(self, yc, pc):
        thr = np.ones(N_CLASSES, np.float32)
        grid = np.linspace(0.05, 2.0, int(self.grid), dtype=np.float32)
        labels = np.arange(N_CLASSES)
        score = lambda t: f1_score(
            yc,
            (pc / t.reshape(1, -1)).argmax(1),
            average="macro",
            zero_division=0,
            labels=labels,
        )
        best = score(thr)
        for _ in range(int(self.passes)):
            moved = False
            for c in range(N_CLASSES):
                best_c = thr[c]
                for t in grid:
                    cand = thr.copy()
                    cand[c] = t
                    s = score(cand)
                    if s > best + 1e-12:
                        best, best_c, moved = s, float(t), True
                thr[c] = best_c
            if not moved:
                break
        return thr.astype(np.float32)

    def predict_proba(self, X):
        p = predict_proba_n(self.clf_, self._stack(X), N_CLASSES, CLASS_IDS)
        p = p / self.thr_.reshape(1, -1)
        return (p / np.clip(p.sum(1, keepdims=True), 1e-12, None)).astype(np.float32)

    def predict(self, X):
        return self.predict_proba(X).argmax(1).astype(np.int64)


class TTAProbe(BaseEstimator, ClassifierMixin):
    """Test-time augmentation wrapper (feature masking) around a base probe.
    Fits clean, perturbs only at predict time."""

    def __init__(
        self,
        base_estimator=None,
        n_aug=8,
        keep_prob=0.9,
        gaussian_std=0.0,
        include_clean=True,
        random_state=42,
    ):
        self.base_estimator = base_estimator
        self.n_aug = n_aug
        self.keep_prob = keep_prob
        self.gaussian_std = gaussian_std
        self.include_clean = include_clean
        self.random_state = random_state

    def _make_base(self):
        if self.base_estimator is None:
            return LinearProbe(
                alpha=3e-5, max_iter=25, grid=15, passes=2, seed=self.random_state
            )
        return clone(self.base_estimator)

    def fit(self, X, y):
        self.base_ = self._make_base()
        self.base_.fit(X, y)
        self.classes_ = getattr(self.base_, "classes_", CLASS_IDS.copy())
        if isinstance(X, dict):
            self.feat_std_ = {
                k: np.asarray(X[k], np.float32).std(0, keepdims=True) + 1e-6 for k in X
            }
        else:
            self.feat_std_ = np.asarray(X, np.float32).std(0, keepdims=True) + 1e-6
        return self

    def _perturb(self, arr, std, rng):
        a = np.asarray(arr, np.float32).copy()
        if self.keep_prob < 1.0:
            mask = (rng.random(a.shape) < self.keep_prob).astype(np.float32)
            a = a * mask / self.keep_prob
        if self.gaussian_std > 0.0:
            a = a + rng.standard_normal(a.shape).astype(np.float32) * (
                self.gaussian_std * std
            )
        return a

    def _perturb_input(self, X, rng):
        if isinstance(X, dict):
            return {k: self._perturb(X[k], self.feat_std_[k], rng) for k in X}
        return self._perturb(X, self.feat_std_, rng)

    def predict_proba(self, X):
        rng = np.random.default_rng(int(self.random_state))
        acc, n = None, 0
        if self.include_clean:
            acc = self.base_.predict_proba(X).astype(np.float32)
            n = 1
        for _ in range(int(self.n_aug)):
            p = self.base_.predict_proba(self._perturb_input(X, rng)).astype(np.float32)
            acc = p if acc is None else acc + p
            n += 1
        p = acc / max(n, 1)
        return (p / np.clip(p.sum(1, keepdims=True), 1e-12, None)).astype(np.float32)

    def predict(self, X):
        return self.predict_proba(X).argmax(1).astype(np.int64)


def _chunk_head_proba(model: Any, X: np.ndarray) -> np.ndarray:
    n = int(X.shape[0])
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X), dtype=np.float32)
        p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
        classes = getattr(model, "classes_", CLASS_IDS)
        out = np.zeros((n, N_CLASSES), dtype=np.float32)
        for j, c in enumerate(classes):
            ci = int(c)
            if 0 <= ci < N_CLASSES:
                out[:, ci] = p[:, j]
    elif hasattr(model, "decision_function"):
        d = model.decision_function(X)
        classes = getattr(
            model, "classes_", CLASS_IDS[: d.shape[1] if d.ndim > 1 else 2]
        )
        scores = np.full((n, N_CLASSES), -20.0, dtype=np.float32)
        if d.ndim == 1:
            d = np.stack([-d, d], axis=1)
        for j, c in enumerate(classes):
            ci = int(c)
            if 0 <= ci < N_CLASSES and j < d.shape[1]:
                scores[:, ci] = d[:, j]
        z = scores.astype(np.float64)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        out = (e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)).astype(
            np.float32
        )
    else:
        pred = model.predict(X).astype(int)
        out = np.full((n, N_CLASSES), 1e-6, dtype=np.float32)
        out[np.arange(n), pred] = 1.0

    row_sum = out.sum(axis=1, keepdims=True)
    bad = ~np.isfinite(row_sum[:, 0]) | (row_sum[:, 0] <= 0)
    out[bad] = 1.0 / N_CLASSES
    out[~bad] /= row_sum[~bad]
    return out


class BratsPath2025ChunkedSGDEnsemble(BaseEstimator, ClassifierMixin):
    """Chunked multi-foundation SGD ensemble. Accepts {fm: array} dicts."""

    def __init__(
        self,
        base_estimator: Any | None = None,
        chunk_size: int = 1024,
        min_chunk_dim: int = 16,
        rare_boost: float = 1.10,
        rare_quantile: float = 0.35,
        rare_classes: Tuple[int, ...] | None = None,
        calibration_fraction: float = 0.10,
        threshold_grid_size: int = 11,
        threshold_passes: int = 1,
        max_train_samples_per_class: int | None = None,
        use_sample_weight: bool = False,
        source_weights: Mapping[str, float] | None = None,
        source_aggregation: str = "head_mean",
        foundation_names: Sequence[str] | None = None,
        foundation_label: str | None = None,
        verbose: int = 0,
        random_state: int = 42,
    ):
        self.base_estimator = base_estimator
        self.chunk_size = chunk_size
        self.min_chunk_dim = min_chunk_dim
        self.rare_boost = rare_boost
        self.rare_quantile = rare_quantile
        self.rare_classes = rare_classes
        self.calibration_fraction = calibration_fraction
        self.threshold_grid_size = threshold_grid_size
        self.threshold_passes = threshold_passes
        self.max_train_samples_per_class = max_train_samples_per_class
        self.use_sample_weight = use_sample_weight
        self.source_weights = source_weights
        self.source_aggregation = source_aggregation
        self.foundation_names = tuple(foundation_names) if foundation_names else tuple()
        self.foundation_label = foundation_label
        self.verbose = verbose
        self.random_state = random_state

    def _iter_chunks(self, dim: int):
        step = int(self.chunk_size)
        made = False
        for start in range(0, int(dim), step):
            stop = min(start + step, int(dim))
            if stop - start >= int(self.min_chunk_dim):
                made = True
                yield int(start), int(stop)
        if not made and int(dim) > 0:
            yield 0, int(dim)

    def _new_estimator(self, seed_offset: int = 0) -> Any:
        est = (
            clone(self.base_estimator)
            if self.base_estimator is not None
            else _sgd(3e-5, "log_loss", seed=int(self.random_state))
        )
        params = getattr(est, "get_params", lambda: {})()
        if "random_state" in params:
            est.set_params(random_state=int(self.random_state) + int(seed_offset))
        return est

    def _source_weight(self, source_name: str) -> float:
        if self.source_weights is None:
            return 1.0
        try:
            return float(self.source_weights.get(source_name, 1.0))
        except AttributeError:
            return 1.0

    def _infer_rare_classes(self, y: np.ndarray) -> np.ndarray:
        if self.rare_classes is not None:
            return np.asarray(self.rare_classes, dtype=np.int64)
        counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=N_CLASSES)
        nonzero = counts[counts > 0]
        if len(nonzero) == 0:
            return np.asarray([], dtype=np.int64)
        cutoff = float(np.quantile(nonzero, float(self.rare_quantile)))
        return np.flatnonzero((counts > 0) & (counts <= cutoff)).astype(np.int64)

    def _subsample_training_indices(self, y: np.ndarray) -> np.ndarray:
        cap = self.max_train_samples_per_class
        if cap is None or int(cap) <= 0:
            return np.arange(len(y), dtype=np.int64)
        cap = int(cap)
        rng = np.random.default_rng(int(self.random_state))
        y = np.asarray(y, dtype=np.int64)
        chosen = []
        for c in range(N_CLASSES):
            idx = np.flatnonzero(y == c)
            if len(idx) == 0:
                continue
            if len(idx) > cap:
                idx = rng.choice(idx, size=cap, replace=False)
            chosen.append(idx.astype(np.int64, copy=False))
        if not chosen:
            return np.arange(len(y), dtype=np.int64)
        return np.sort(np.concatenate(chosen).astype(np.int64, copy=False))

    def _split_fit_calibration(
        self, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        frac = float(self.calibration_fraction)
        if not (0.0 < frac < 0.5):
            return np.arange(len(y), dtype=np.int64), None
        counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=N_CLASSES)
        if len(counts[counts > 0]) < 2 or counts[counts > 0].min() < 2:
            return np.arange(len(y), dtype=np.int64), None
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=frac, random_state=int(self.random_state)
        )
        idx = np.arange(len(y), dtype=np.int64)
        try:
            fit_rel, cal_rel = next(sss.split(np.zeros(len(y)), y))
            return idx[fit_rel], idx[cal_rel]
        except Exception:
            return idx, None

    def _fit_heads(
        self,
        X: Mapping[str, np.ndarray],
        y_by_source: Mapping[str, np.ndarray],
    ):
        """Fit each source's chunk heads on that source's own training rows.

        Base OOF rows remain aligned across sources. Train-only augmentations
        such as Ivy GAP are allowed to be source-specific: each source can have
        its own extra rows and labels, and source_mean/head_mean aggregation does
        not require extra-row alignment.
        """
        if set(X) != set(y_by_source):
            raise ValueError(
                f"X/y source keys differ: X={sorted(X)}, y={sorted(y_by_source)}"
            )

        jobs = []
        for source_name in sorted(X.keys()):
            X_source = np.asarray(X[source_name], dtype=np.float32)
            y_source = np.asarray(y_by_source[source_name], dtype=np.int64)
            if X_source.ndim != 2 or len(X_source) != len(y_source):
                raise ValueError(
                    f"{source_name}: X/y shape mismatch: X={X_source.shape}, y={y_source.shape}"
                )
            dim = X_source.shape[1]
            for start, stop in self._iter_chunks(dim):
                jobs.append((source_name, int(start), int(stop)))

        if not jobs:
            raise ValueError(
                "No chunk heads were created; check embedding dims / chunk_size."
            )

        source_rows = {name: int(len(y_by_source[name])) for name in sorted(X)}
        iterator = (
            tqdm(
                jobs,
                desc=f"fit chunk-SGD heads n={len(jobs)} source_rows={source_rows}",
                leave=False,
            )
            if int(self.verbose) > 0
            else jobs
        )

        heads = []
        for seed_offset, (source_name, start, stop) in enumerate(iterator):
            Xchunk = np.asarray(X[source_name], dtype=np.float32)[:, start:stop]
            y_source = np.asarray(y_by_source[source_name], dtype=np.int64)
            sw = (
                compute_sample_weight("balanced", y_source)
                if bool(self.use_sample_weight)
                else None
            )
            clf = self._new_estimator(seed_offset=seed_offset)
            try:
                if sw is None:
                    clf.fit(Xchunk, y_source)
                else:
                    try:
                        clf.fit(Xchunk, y_source, sample_weight=sw)
                    except TypeError:
                        clf.fit(Xchunk, y_source)
            except ValueError as exc:
                params = getattr(clf, "get_params", lambda: {})()
                if params.get("early_stopping", False) and hasattr(clf, "set_params"):
                    clf = self._new_estimator(seed_offset=seed_offset)
                    clf.set_params(early_stopping=False)
                    if sw is None:
                        clf.fit(Xchunk, y_source)
                    else:
                        try:
                            clf.fit(Xchunk, y_source, sample_weight=sw)
                        except TypeError:
                            clf.fit(Xchunk, y_source)
                else:
                    raise exc
            heads.append((source_name, int(start), int(stop), clf))
        self.heads_ = heads
        return self

    def _raw_avg_proba(self, X: Mapping[str, np.ndarray]) -> np.ndarray:
        n = int(next(iter(X.values())).shape[0])
        aggregation = str(self.source_aggregation)

        if aggregation == "head_mean":
            acc = np.zeros((n, N_CLASSES), dtype=np.float32)
            denom = 0.0
            for source_name, start, stop, clf in self.heads_:
                w = max(self._source_weight(source_name), 0.0)
                if w == 0.0:
                    continue
                Xchunk = np.asarray(X[source_name], dtype=np.float32)[:, start:stop]
                acc += w * _chunk_head_proba(clf, Xchunk)
                denom += w
            acc /= max(denom, 1e-12)
            acc /= np.clip(acc.sum(axis=1, keepdims=True), 1e-12, None)
            return acc

        if aggregation != "source_mean":
            raise ValueError(f"Unknown source_aggregation={aggregation!r}")

        per_source_acc: Dict[str, np.ndarray] = {}
        per_source_n: Dict[str, int] = {}
        for source_name, start, stop, clf in self.heads_:
            if source_name not in per_source_acc:
                per_source_acc[source_name] = np.zeros((n, N_CLASSES), dtype=np.float32)
                per_source_n[source_name] = 0
            Xchunk = np.asarray(X[source_name], dtype=np.float32)[:, start:stop]
            per_source_acc[source_name] += _chunk_head_proba(clf, Xchunk)
            per_source_n[source_name] += 1

        acc = np.zeros((n, N_CLASSES), dtype=np.float32)
        denom = 0.0
        for source_name in sorted(per_source_acc.keys()):
            w = max(self._source_weight(source_name), 0.0)
            if w == 0.0:
                continue
            p_src = per_source_acc[source_name] / max(per_source_n[source_name], 1)
            p_src /= np.clip(p_src.sum(axis=1, keepdims=True), 1e-12, None)
            acc += w * p_src
            denom += w
        acc /= max(denom, 1e-12)
        acc /= np.clip(acc.sum(axis=1, keepdims=True), 1e-12, None)
        return acc.astype(np.float32, copy=False)

    def _apply_rare_boost(self, proba: np.ndarray) -> np.ndarray:
        out = proba.astype(np.float32, copy=True)
        if float(self.rare_boost) != 1.0 and len(getattr(self, "rare_classes_", [])):
            out[:, self.rare_classes_] *= float(self.rare_boost)
            out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
        return out

    @staticmethod
    def _predict_with_thresholds(
        proba: np.ndarray, thresholds: np.ndarray
    ) -> np.ndarray:
        return (
            (proba / np.clip(thresholds.reshape(1, -1), 1e-6, None))
            .argmax(axis=1)
            .astype(np.int64)
        )

    def _optimize_thresholds(self, y_cal: np.ndarray, p_cal: np.ndarray) -> np.ndarray:
        y_cal = np.asarray(y_cal, dtype=np.int64)
        thresholds = np.ones(N_CLASSES, dtype=np.float32)
        grid = np.linspace(0.05, 2.00, int(self.threshold_grid_size), dtype=np.float32)
        best = f1_score(
            y_cal,
            self._predict_with_thresholds(p_cal, thresholds),
            average="macro",
            zero_division=0,
        )
        for _ in range(int(self.threshold_passes)):
            improved = False
            for c in range(N_CLASSES):
                best_c = float(thresholds[c])
                for t in grid:
                    cand = thresholds.copy()
                    cand[c] = float(t)
                    score = f1_score(
                        y_cal,
                        self._predict_with_thresholds(p_cal, cand),
                        average="macro",
                        zero_division=0,
                    )
                    if score > best + 1e-12:
                        best = float(score)
                        best_c = float(t)
                        improved = True
                thresholds[c] = best_c
            if not improved:
                break
        return thresholds.astype(np.float32)

    def fit(
        self,
        X: Mapping[str, np.ndarray],
        y: np.ndarray,
        source_train_extras: Mapping[str, Tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        """Fit on aligned BraTS rows, optionally with train-only extras.

        The base BraTS split still drives rare-class detection and threshold
        calibration. ``source_train_extras`` is used only for chunk-head fitting;
        it is never visible to the BraTS-only calibration partition and is never
        part of validation OOF. This is the same Ivy-GAP protocol used by the
        train/inference script.
        """
        if not isinstance(X, Mapping):
            raise TypeError("Expected a dict {foundation_name: np.ndarray}.")
        y = np.asarray(y, dtype=np.int64)
        if not X:
            raise ValueError("X must contain at least one foundation.")
        if any(len(np.asarray(v)) != len(y) for v in X.values()):
            raise ValueError(
                "All aligned BraTS foundation arrays must have len(X)==len(y)."
            )

        extras = dict(source_train_extras or {})
        unknown = set(extras) - set(X)
        if unknown:
            raise KeyError(f"Extras supplied for absent source(s): {sorted(unknown)}")

        self.classes_ = CLASS_IDS.copy()
        self.source_names_ = tuple(sorted(X.keys()))
        self.rare_classes_ = self._infer_rare_classes(y)

        pool_idx = self._subsample_training_indices(y)
        if int(self.verbose) > 0 and len(pool_idx) < len(y):
            counts = np.bincount(y[pool_idx], minlength=N_CLASSES)
            nonzero = {int(c): int(n) for c, n in enumerate(counts) if n > 0}
            print(
                f"[chunk-sgd] class-balanced train cap: {len(pool_idx):,}/{len(y):,}, counts={nonzero}"
            )

        y_pool = y[pool_idx]
        fit_rel, cal_rel = self._split_fit_calibration(y_pool)
        fit_idx = pool_idx[fit_rel]
        cal_idx = None if cal_rel is None else pool_idx[cal_rel]

        X_fit_by_source: Dict[str, np.ndarray] = {}
        y_fit_by_source: Dict[str, np.ndarray] = {}
        self.source_extra_rows_ = {}
        for source_name, X_source in X.items():
            X_base = np.asarray(X_source, dtype=np.float32)[fit_idx]
            y_base = y[fit_idx]
            if source_name in extras:
                X_extra, y_extra = extras[source_name]
                X_extra = np.asarray(X_extra, dtype=np.float32)
                y_extra = np.asarray(y_extra, dtype=np.int64)
                if X_extra.ndim != 2 or X_extra.shape[1] != X_base.shape[1]:
                    raise ValueError(
                        f"{source_name}: extra feature shape {X_extra.shape} is incompatible "
                        f"with base shape {X_base.shape}."
                    )
                if len(X_extra) != len(y_extra):
                    raise ValueError(
                        f"{source_name}: extra X/y lengths disagree: {len(X_extra)} vs {len(y_extra)}."
                    )
                if not np.isfinite(X_extra).all():
                    raise ValueError(f"{source_name}: extras contain NaN/Inf.")
                X_fit_by_source[source_name] = np.concatenate([X_base, X_extra], axis=0)
                y_fit_by_source[source_name] = np.concatenate([y_base, y_extra], axis=0)
                self.source_extra_rows_[source_name] = int(len(y_extra))
            else:
                X_fit_by_source[source_name] = X_base
                y_fit_by_source[source_name] = y_base
                self.source_extra_rows_[source_name] = 0

        self.source_fit_rows_ = {
            source_name: int(len(y_fit_by_source[source_name]))
            for source_name in self.source_names_
        }
        self._fit_heads(X_fit_by_source, y_fit_by_source)

        if cal_idx is not None and len(cal_idx) > 0:
            X_cal = {k: np.asarray(v, dtype=np.float32)[cal_idx] for k, v in X.items()}
            p_cal = self._apply_rare_boost(self._raw_avg_proba(X_cal))
            self.thresholds_ = self._optimize_thresholds(y[cal_idx], p_cal)
            del X_cal, p_cal
        else:
            self.thresholds_ = np.ones(N_CLASSES, dtype=np.float32)

        del X_fit_by_source, y_fit_by_source, y_pool, pool_idx
        gc.collect()
        return self

    def predict_proba(self, X: Mapping[str, np.ndarray]) -> np.ndarray:
        p = self._apply_rare_boost(self._raw_avg_proba(X))
        p = p / np.clip(self.thresholds_.reshape(1, -1), 1e-6, None)
        p /= np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
        return p.astype(np.float32, copy=False)

    def predict(self, X: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1).astype(np.int64)


def _make_feature_masks(
    dim: int,
    n_masks: int,
    mask_fraction: float,
    min_mask_dim: int,
    scheme: str,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    dim = int(dim)
    n = int(n_masks)
    m = int(round(float(mask_fraction) * dim))
    m = max(int(min_mask_dim), m)
    m = min(m, dim)

    if m >= dim:
        return [np.arange(dim, dtype=np.int64) for _ in range(n)]

    if str(scheme) == "random":
        masks = [
            np.sort(rng.choice(dim, size=m, replace=False)).astype(np.int64)
            for _ in range(n)
        ]
    elif str(scheme) == "balanced":
        total = n * m
        reps = int(np.ceil(total / dim))
        pool = np.concatenate([rng.permutation(dim) for _ in range(reps)])[:total]
        masks = [
            np.unique(pool[i * m : (i + 1) * m]).astype(np.int64) for i in range(n)
        ]
    else:
        raise ValueError(f"Unknown mask scheme {scheme!r}")

    covered = np.zeros(dim, dtype=bool)
    for mk in masks:
        covered[mk] = True
    for f in np.flatnonzero(~covered):
        k = int(rng.integers(len(masks)))
        masks[k] = np.union1d(masks[k], np.int64(f)).astype(np.int64)
    return masks


class FeatureMaskedSGDEnsemble(BaseEstimator, ClassifierMixin):
    """Feature-masking sibling of BratsPath2025ChunkedSGDEnsemble: same dict
    API / aggregation / calibration, heads trained on random feature masks
    instead of contiguous chunks."""

    def __init__(
        self,
        base_estimator: Any | None = None,
        n_masks_per_source: int = 12,
        mask_fraction: float = 0.5,
        min_mask_dim: int = 16,
        mask_scheme: str = "balanced",
        rare_boost: float = 1.10,
        rare_quantile: float = 0.35,
        rare_classes: Tuple[int, ...] | None = None,
        calibration_fraction: float = 0.10,
        threshold_grid_size: int = 11,
        threshold_passes: int = 1,
        max_train_samples_per_class: int | None = None,
        use_sample_weight: bool = False,
        source_weights: Mapping[str, float] | None = None,
        source_aggregation: str = "head_mean",
        foundation_names: Sequence[str] | None = None,
        foundation_label: str | None = None,
        verbose: int = 0,
        random_state: int = 42,
    ):
        self.base_estimator = base_estimator
        self.n_masks_per_source = n_masks_per_source
        self.mask_fraction = mask_fraction
        self.min_mask_dim = min_mask_dim
        self.mask_scheme = mask_scheme
        self.rare_boost = rare_boost
        self.rare_quantile = rare_quantile
        self.rare_classes = rare_classes
        self.calibration_fraction = calibration_fraction
        self.threshold_grid_size = threshold_grid_size
        self.threshold_passes = threshold_passes
        self.max_train_samples_per_class = max_train_samples_per_class
        self.use_sample_weight = use_sample_weight
        self.source_weights = source_weights
        self.source_aggregation = source_aggregation
        self.foundation_names = tuple(foundation_names) if foundation_names else tuple()
        self.foundation_label = foundation_label
        self.verbose = verbose
        self.random_state = random_state

    def _new_estimator(self, seed_offset: int = 0) -> Any:
        est = (
            clone(self.base_estimator)
            if self.base_estimator is not None
            else _sgd(3e-5, "log_loss", seed=int(self.random_state))
        )
        params = getattr(est, "get_params", lambda: {})()
        if "random_state" in params:
            est.set_params(random_state=int(self.random_state) + int(seed_offset))
        return est

    def _source_weight(self, source_name: str) -> float:
        if self.source_weights is None:
            return 1.0
        try:
            return float(self.source_weights.get(source_name, 1.0))
        except AttributeError:
            return 1.0

    def _infer_rare_classes(self, y: np.ndarray) -> np.ndarray:
        if self.rare_classes is not None:
            return np.asarray(self.rare_classes, dtype=np.int64)
        counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=N_CLASSES)
        nonzero = counts[counts > 0]
        if len(nonzero) == 0:
            return np.asarray([], dtype=np.int64)
        cutoff = float(np.quantile(nonzero, float(self.rare_quantile)))
        return np.flatnonzero((counts > 0) & (counts <= cutoff)).astype(np.int64)

    def _subsample_training_indices(self, y: np.ndarray) -> np.ndarray:
        cap = self.max_train_samples_per_class
        if cap is None or int(cap) <= 0:
            return np.arange(len(y), dtype=np.int64)
        cap = int(cap)
        rng = np.random.default_rng(int(self.random_state))
        y = np.asarray(y, dtype=np.int64)
        chosen = []
        for c in range(N_CLASSES):
            idx = np.flatnonzero(y == c)
            if len(idx) == 0:
                continue
            if len(idx) > cap:
                idx = rng.choice(idx, size=cap, replace=False)
            chosen.append(idx.astype(np.int64, copy=False))
        if not chosen:
            return np.arange(len(y), dtype=np.int64)
        return np.sort(np.concatenate(chosen).astype(np.int64, copy=False))

    def _split_fit_calibration(
        self, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        frac = float(self.calibration_fraction)
        if not (0.0 < frac < 0.5):
            return np.arange(len(y), dtype=np.int64), None
        counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=N_CLASSES)
        if len(counts[counts > 0]) < 2 or counts[counts > 0].min() < 2:
            return np.arange(len(y), dtype=np.int64), None
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=frac, random_state=int(self.random_state)
        )
        idx = np.arange(len(y), dtype=np.int64)
        try:
            fit_rel, cal_rel = next(sss.split(np.zeros(len(y)), y))
            return idx[fit_rel], idx[cal_rel]
        except Exception:
            return idx, None

    def _fit_one(self, Xm: np.ndarray, y: np.ndarray, sw, seed_offset: int) -> Any:
        clf = self._new_estimator(seed_offset=seed_offset)
        try:
            if sw is None:
                clf.fit(Xm, y)
            else:
                try:
                    clf.fit(Xm, y, sample_weight=sw)
                except TypeError:
                    clf.fit(Xm, y)
        except ValueError as exc:
            params = getattr(clf, "get_params", lambda: {})()
            if params.get("early_stopping", False) and hasattr(clf, "set_params"):
                clf = self._new_estimator(seed_offset=seed_offset)
                clf.set_params(early_stopping=False)
                if sw is None:
                    clf.fit(Xm, y)
                else:
                    try:
                        clf.fit(Xm, y, sample_weight=sw)
                    except TypeError:
                        clf.fit(Xm, y)
            else:
                raise exc
        return clf

    def _fit_heads(self, X: Mapping[str, np.ndarray], y: np.ndarray):
        y = np.asarray(y, dtype=np.int64)
        sw = (
            compute_sample_weight("balanced", y)
            if bool(self.use_sample_weight)
            else None
        )

        jobs: List[Tuple[str, np.ndarray]] = []
        for si, source_name in enumerate(sorted(X.keys())):
            dim = int(X[source_name].shape[1])
            rng = np.random.default_rng(int(self.random_state) + 1000 * (si + 1))
            for mask in _make_feature_masks(
                dim,
                self.n_masks_per_source,
                self.mask_fraction,
                self.min_mask_dim,
                self.mask_scheme,
                rng,
            ):
                jobs.append((source_name, mask))
        if not jobs:
            raise ValueError("No feature-mask heads were created.")
        iterator = (
            tqdm(
                jobs,
                desc=f"fit feature-mask heads n={len(jobs)} rows={len(y):,}",
                leave=False,
            )
            if int(self.verbose) > 0
            else jobs
        )

        heads = []
        for seed_offset, (source_name, mask) in enumerate(iterator):
            Xm = np.asarray(X[source_name], dtype=np.float32)[:, mask]
            clf = self._fit_one(Xm, y, sw, seed_offset)
            heads.append((source_name, mask, clf))
        self.heads_ = heads
        return self

    def _raw_avg_proba(self, X: Mapping[str, np.ndarray]) -> np.ndarray:
        n = int(next(iter(X.values())).shape[0])
        aggregation = str(self.source_aggregation)

        if aggregation == "head_mean":
            acc = np.zeros((n, N_CLASSES), dtype=np.float32)
            denom = 0.0
            for source_name, mask, clf in self.heads_:
                w = max(self._source_weight(source_name), 0.0)
                if w == 0.0:
                    continue
                Xm = np.asarray(X[source_name], dtype=np.float32)[:, mask]
                acc += w * _chunk_head_proba(clf, Xm)
                denom += w
            acc /= max(denom, 1e-12)
            acc /= np.clip(acc.sum(axis=1, keepdims=True), 1e-12, None)
            return acc

        if aggregation != "source_mean":
            raise ValueError(f"Unknown source_aggregation={aggregation!r}")

        per_source_acc: Dict[str, np.ndarray] = {}
        per_source_n: Dict[str, int] = {}
        for source_name, mask, clf in self.heads_:
            if source_name not in per_source_acc:
                per_source_acc[source_name] = np.zeros((n, N_CLASSES), dtype=np.float32)
                per_source_n[source_name] = 0
            Xm = np.asarray(X[source_name], dtype=np.float32)[:, mask]
            per_source_acc[source_name] += _chunk_head_proba(clf, Xm)
            per_source_n[source_name] += 1

        acc = np.zeros((n, N_CLASSES), dtype=np.float32)
        denom = 0.0
        for source_name in sorted(per_source_acc.keys()):
            w = max(self._source_weight(source_name), 0.0)
            if w == 0.0:
                continue
            p_src = per_source_acc[source_name] / max(per_source_n[source_name], 1)
            p_src /= np.clip(p_src.sum(axis=1, keepdims=True), 1e-12, None)
            acc += w * p_src
            denom += w
        acc /= max(denom, 1e-12)
        acc /= np.clip(acc.sum(axis=1, keepdims=True), 1e-12, None)
        return acc.astype(np.float32, copy=False)

    def _apply_rare_boost(self, proba: np.ndarray) -> np.ndarray:
        out = proba.astype(np.float32, copy=True)
        if float(self.rare_boost) != 1.0 and len(getattr(self, "rare_classes_", [])):
            out[:, self.rare_classes_] *= float(self.rare_boost)
            out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
        return out

    @staticmethod
    def _predict_with_thresholds(
        proba: np.ndarray, thresholds: np.ndarray
    ) -> np.ndarray:
        return (
            (proba / np.clip(thresholds.reshape(1, -1), 1e-6, None))
            .argmax(axis=1)
            .astype(np.int64)
        )

    def _optimize_thresholds(self, y_cal: np.ndarray, p_cal: np.ndarray) -> np.ndarray:
        y_cal = np.asarray(y_cal, dtype=np.int64)
        thresholds = np.ones(N_CLASSES, dtype=np.float32)
        grid = np.linspace(0.05, 2.00, int(self.threshold_grid_size), dtype=np.float32)
        best = f1_score(
            y_cal,
            self._predict_with_thresholds(p_cal, thresholds),
            average="macro",
            zero_division=0,
        )
        for _ in range(int(self.threshold_passes)):
            improved = False
            for c in range(N_CLASSES):
                best_c = float(thresholds[c])
                for t in grid:
                    cand = thresholds.copy()
                    cand[c] = float(t)
                    score = f1_score(
                        y_cal,
                        self._predict_with_thresholds(p_cal, cand),
                        average="macro",
                        zero_division=0,
                    )
                    if score > best + 1e-12:
                        best = float(score)
                        best_c = float(t)
                        improved = True
                thresholds[c] = best_c
            if not improved:
                break
        return thresholds.astype(np.float32)

    def fit(self, X: Mapping[str, np.ndarray], y: np.ndarray):
        if not isinstance(X, Mapping):
            raise TypeError("Expected a dict {foundation_name: np.ndarray}.")
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = CLASS_IDS.copy()
        self.source_names_ = tuple(sorted(X.keys()))
        self.rare_classes_ = self._infer_rare_classes(y)

        pool_idx = self._subsample_training_indices(y)
        y_pool = y[pool_idx]
        fit_rel, cal_rel = self._split_fit_calibration(y_pool)
        fit_idx = pool_idx[fit_rel]
        cal_idx = None if cal_rel is None else pool_idx[cal_rel]

        X_fit = {k: np.asarray(v, dtype=np.float32)[fit_idx] for k, v in X.items()}
        self._fit_heads(X_fit, y[fit_idx])

        if cal_idx is not None and len(cal_idx) > 0:
            X_cal = {k: np.asarray(v, dtype=np.float32)[cal_idx] for k, v in X.items()}
            p_cal = self._apply_rare_boost(self._raw_avg_proba(X_cal))
            self.thresholds_ = self._optimize_thresholds(y[cal_idx], p_cal)
            del X_cal, p_cal
        else:
            self.thresholds_ = np.ones(N_CLASSES, dtype=np.float32)

        del X_fit, y_pool, pool_idx
        gc.collect()
        return self

    def predict_proba(self, X: Mapping[str, np.ndarray]) -> np.ndarray:
        p = self._apply_rare_boost(self._raw_avg_proba(X))
        p = p / np.clip(self.thresholds_.reshape(1, -1), 1e-6, None)
        p /= np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
        return p.astype(np.float32, copy=False)

    def predict(self, X: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1).astype(np.int64)


# -----------------------------------------------------------------------------
# Optional augmentation-set materialization
# -----------------------------------------------------------------------------


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{__import__('os').getpid()}")
    np.savez(tmp, **arrays)
    tmp_npz = Path(str(tmp) if str(tmp).endswith(".npz") else str(tmp) + ".npz")
    tmp_npz.replace(path)


def atomic_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{__import__('os').getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _source_artifact(foundation: str, suffix: str) -> str:
    """Map a short suffix like 'stainaug_local' to '<foundation>_stainaug_local'.

    If suffix is already expanded, e.g. 'virchow2_stainaug_local', it is left
    as-is for path lookup.
    """
    return suffix if suffix.startswith(f"{foundation}_") else f"{foundation}_{suffix}"


def _aug_name_tag(foundation: str, suffix: str) -> str:
    """Return the string prefix stored in patch names for one augmentation set.

    Unlike the single-foundation smoke-test script, this deliberately strips the
    leading '<foundation>_' when an expanded artifact name is supplied. That way
    a composite such as ('virchow2', 'hoptimus1') can intersect augmented rows by
    a common name like 'stainaug::patch#aug3' instead of dropping them because one
    source wrote 'virchow2_stainaug::...' and another wrote
    'hoptimus1_stainaug::...'.
    """
    artifact = _source_artifact(foundation, suffix)
    prefix = f"{foundation}_"
    return artifact[len(prefix) :] if artifact.startswith(prefix) else artifact


def _safe_token(text: str) -> str:
    out = []
    for ch in str(text):
        out.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    token = "".join(out).strip("_")
    return token or "aug"


def _merged_foundation_name(
    foundation: str, suffixes: Tuple[str, ...], partial_suffix: str
) -> str:
    token = "__".join(_safe_token(_aug_name_tag(foundation, x)) for x in suffixes)
    if len(token) > 80:
        token = hashlib.blake2b(token.encode("utf-8"), digest_size=6).hexdigest()
    return f"{foundation}_{partial_suffix}_{token}"


def consolidate_partial_aug(parts_root: Path) -> Dict[Tuple[str, str], List[Path]]:
    """Group completed in-progress part files by (patient-dir, slide-stem)."""
    groups: Dict[Tuple[str, str], List[Path]] = {}
    for shard_dir in sorted(parts_root.glob("shard-*")):
        if not (shard_dir / "_SHARD_DONE.json").exists():
            continue
        for part in shard_dir.glob("patient-*/slide-*.part.npz"):
            key = (part.parent.name, part.stem.replace(".part", ""))
            groups.setdefault(key, []).append(part)
    return groups


def consolidate_final_aug(final_root: Path) -> Dict[Tuple[str, str], List[Path]]:
    """Group already-consolidated augmented files by (patient-dir, slide-stem)."""
    groups: Dict[Tuple[str, str], List[Path]] = {}
    for fp in sorted(final_root.glob("patient-*/slide-*.npz")):
        key = (fp.parent.name, fp.stem)
        groups.setdefault(key, []).append(fp)
    return groups


def load_aug_file_group(
    paths: List[Path], name_tag: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, ys, names = [], [], []
    for p in sorted(paths):
        d = np.load(p, allow_pickle=True)
        Xs.append(d["X"])
        ys.append(d["y"])
        names.extend(d["names"].tolist())
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    names_arr = np.asarray(names, dtype=object)

    # Dedup within this augmentation source by original name (keep last), stable
    # sort by name. Across different suffixes, keep both views via name_tag.
    order = np.argsort(names_arr.astype(str), kind="mergesort")
    X, y, names_arr = X[order], y[order], names_arr[order]
    if len(names_arr) > 1:
        _, ulr = np.unique(names_arr[::-1].astype(str), return_index=True)
        keep = np.sort(len(names_arr) - 1 - ulr)
        X, y, names_arr = X[keep], y[keep], names_arr[keep]

    names_arr = np.asarray(
        [f"{name_tag}::{n}" for n in names_arr.astype(str)], dtype=object
    )
    return X, y, names_arr


def find_aug_groups(
    foundation: str, suffix: str
) -> Tuple[str, str, Path, Dict[Tuple[str, str], List[Path]]]:
    """Return augmented files for one configured suffix.

    Search order:
      1. completed in-progress shards under artifacts/embedding_parts/<artifact>/<split>
      2. already-consolidated files under embeddings_by_patient_slide/<artifact>/<split>
    """
    artifact = _source_artifact(foundation, suffix)
    name_tag = _aug_name_tag(foundation, suffix)
    parts_root = SWEEP.artifacts / "embedding_parts" / artifact / SWEEP.split
    final_root = SWEEP.embedding_root / artifact / SWEEP.split

    if parts_root.exists():
        groups = consolidate_partial_aug(parts_root)
        if groups:
            return artifact, name_tag, parts_root, groups

    if final_root.exists():
        groups = consolidate_final_aug(final_root)
        if groups:
            return artifact, name_tag, final_root, groups

    raise FileNotFoundError(
        f"No completed augmented embeddings found for suffix={suffix!r} / artifact={artifact!r}.\n"
        f"Looked in:\n"
        f"  - {parts_root}\n"
        f"  - {final_root}"
    )


def build_augmented_foundation(
    foundation: str, aug_artifact_suffixes: Tuple[str, ...]
) -> str:
    """Build/rebuild an identity + augmentation merged foundation.

    The returned foundation directory is written under SWEEP.embedding_root and
    can be consumed by the normal OOF harness exactly like an ordinary foundation.
    For composites, augmented row names are tagged by suffix rather than by full
    foundation artifact so rows can align across different foundations.
    """
    aug_artifact_suffixes = tuple(aug_artifact_suffixes)
    if not aug_artifact_suffixes:
        return foundation

    ident_root = SWEEP.embedding_root / foundation / SWEEP.split
    if not ident_root.exists():
        raise FileNotFoundError(f"Identity embeddings not found: {ident_root}")

    merged_name = _merged_foundation_name(
        foundation, aug_artifact_suffixes, SWEEP.partial_suffix
    )
    final_root = SWEEP.embedding_root / merged_name / SWEEP.split
    if final_root.exists():
        shutil.rmtree(final_root)
    final_root.mkdir(parents=True, exist_ok=True)

    source_groups: Dict[str, Dict[Tuple[str, str], List[Path]]] = {}
    source_name_tags: Dict[str, str] = {}
    source_roots: Dict[str, str] = {}
    for suffix in aug_artifact_suffixes:
        artifact, name_tag, root, groups = find_aug_groups(foundation, suffix)
        source_groups[artifact] = groups
        source_name_tags[artifact] = name_tag
        source_roots[artifact] = str(root)
        print(
            f"[aug-source:{foundation}] {artifact}: {len(groups)} (patient, slide) files from {root}"
        )

    n_with_any_aug = n_ident_only = 0
    n_aug_rows_by_source: Dict[str, int] = {artifact: 0 for artifact in source_groups}
    n_files_by_source: Dict[str, int] = {artifact: 0 for artifact in source_groups}

    files = sorted(ident_root.glob("patient-*/slide-*.npz"))
    if SWEEP.max_files is not None:
        files = files[: SWEEP.max_files]
    for f in tqdm(files, desc=f"merge identity + aug for {foundation}"):
        pdir, sstem = f.parent.name, f.stem
        d = np.load(f, allow_pickle=True)
        X, y, names = d["X"], d["y"], d["names"]

        key = (pdir, sstem)
        got_aug_for_file = False
        for artifact, groups in source_groups.items():
            if key not in groups:
                continue
            aX, ay, anames = load_aug_file_group(
                groups[key], name_tag=source_name_tags[artifact]
            )
            X = np.concatenate([X, aX.astype(X.dtype, copy=False)], axis=0)
            y = np.concatenate([y, ay.astype(y.dtype, copy=False)], axis=0)
            names = np.concatenate([names, anames], axis=0)
            n_aug_rows_by_source[artifact] += int(aX.shape[0])
            n_files_by_source[artifact] += 1
            got_aug_for_file = True

        if got_aug_for_file:
            n_with_any_aug += 1
        else:
            n_ident_only += 1

        out = final_root / pdir / f"{sstem}.npz"
        atomic_npz(
            out,
            X=X,
            y=y,
            names=names,
            patient=np.asarray(pdir, dtype=object),
            slide=np.asarray(sstem, dtype=object),
        )

    atomic_json(
        final_root / "manifest.json",
        {
            "foundation": merged_name,
            "base_foundation": foundation,
            "split": SWEEP.split,
            "augmentation_artifact_suffixes": list(aug_artifact_suffixes),
            "augmentation_name_tags": source_name_tags,
            "augmentation_source_roots": source_roots,
            "note": "identity + configured augmentation artifacts; generated by probe_sweep_oof_with_aug.py",
            "n_files_with_any_aug": n_with_any_aug,
            "n_files_identity_only": n_ident_only,
            "n_aug_rows_by_source": n_aug_rows_by_source,
            "n_files_by_source": n_files_by_source,
            "max_files": SWEEP.max_files,
        },
    )
    print(
        f"[merged-aug:{foundation}] {n_with_any_aug} slide files got augmented rows, {n_ident_only} were identity-only"
    )
    for artifact, n in n_aug_rows_by_source.items():
        print(f"[merged-aug:{foundation}] {artifact}: {n:,} augmented rows")
    print(f"[merged-aug:{foundation}] foundation ready: {merged_name}")
    return merged_name


def materialize_foundation(
    foundation: str,
    aug_artifact_suffixes: Tuple[str, ...],
    cache: Dict[Tuple[str, Tuple[str, ...]], str],
) -> str:
    suffixes = tuple(aug_artifact_suffixes)
    if not suffixes:
        return foundation
    key = (foundation, suffixes)
    if key not in cache:
        cache[key] = build_augmented_foundation(foundation, suffixes)
    return cache[key]


def composite_suffixes_for(
    cc: CompositeProbeConfig, foundation: str
) -> Tuple[str, ...]:
    if foundation in cc.per_foundation_aug_artifact_suffixes:
        return tuple(cc.per_foundation_aug_artifact_suffixes[foundation])
    return tuple(cc.aug_artifact_suffixes)


IVY_AUG_TOKEN = "ivy"
IVY_SUPPORTED_CLASS_IDS = (0, 2, 4, 5, 7)  # CT, IC, MP, NC, PN
IVY_SOURCE_ROOT_ATTRS: Mapping[str, str] = {
    "virchow2": "ivy_virchow2_root",
    "hoptimus1": "ivy_hoptimus1_root",
}


def has_ivy_aug(suffixes: Tuple[str, ...]) -> bool:
    return any(str(s).lower() == IVY_AUG_TOKEN for s in suffixes)


def non_ivy_aug_suffixes(suffixes: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(s for s in suffixes if str(s).lower() != IVY_AUG_TOKEN)


def label_with_train_augmentations(
    foundation_label: str, suffixes: Tuple[str, ...]
) -> str:
    tags = []
    if has_ivy_aug(suffixes):
        tags.append("ivy")
    return (
        foundation_label
        if not tags
        else f"{foundation_label}__trainaug_{'+'.join(tags)}"
    )


def _norm_summary(X: np.ndarray, max_rows: int = 20_000) -> Dict[str, float]:
    """Compact, deterministic row-L2 norm summary for representation matching."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError(f"Expected a non-empty 2D embedding array, got {X.shape}")
    if len(X) > int(max_rows):
        idx = np.linspace(0, len(X) - 1, int(max_rows), dtype=np.int64)
        X = X[idx]
    norms = np.linalg.norm(X, axis=1)
    return {
        "n_sample": int(len(norms)),
        "p05": float(np.quantile(norms, 0.05)),
        "median": float(np.median(norms)),
        "p95": float(np.quantile(norms, 0.95)),
    }


def _looks_unit_normed(stats: Mapping[str, float]) -> bool:
    """Conservative detection of row-wise L2-normalized embeddings."""
    return (
        0.95 <= float(stats["p05"]) <= 1.05
        and 0.98 <= float(stats["median"]) <= 1.02
        and 0.95 <= float(stats["p95"]) <= 1.05
    )


def _row_l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization in float32, preserving all-zero rows."""
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    out = X.copy()
    nonzero = norms[:, 0] > 0
    out[nonzero] /= norms[nonzero]
    return out


def _ivy_root_for_source(source_name: str) -> Path:
    attr = IVY_SOURCE_ROOT_ATTRS.get(source_name)
    if attr is None:
        raise KeyError(f"No Ivy root is configured for source {source_name!r}.")
    return Path(getattr(SWEEP, attr))


def _read_ivy_embedding_config(
    ivy_root: Path, expected_foundation: str
) -> Dict[str, Any]:
    path = ivy_root.expanduser().resolve() / "embedding_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Ivy embedding configuration: {path}")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot parse Ivy embedding configuration: {path}") from exc

    actual = str(cfg.get("foundation", ""))
    if actual != expected_foundation:
        raise RuntimeError(
            f"{path}: foundation={actual!r}; expected {expected_foundation!r}."
        )
    dim = cfg.get("embedding_dim")
    if not isinstance(dim, int) or dim <= 0:
        raise RuntimeError(f"{path}: invalid embedding_dim={dim!r}.")
    return cfg


def _load_ivy_foundation_blocks(
    ivy_root: Path,
    expected_foundation: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load one raw Ivy source tree, keeping one unique row per patch UID."""
    ivy_root = ivy_root.expanduser().resolve()
    cfg = _read_ivy_embedding_config(ivy_root, expected_foundation)
    expected_dim = int(cfg["embedding_dim"])
    block_files = sorted((ivy_root / "blocks").glob("block-*.npz"))
    if not block_files:
        raise RuntimeError(f"No Ivy blocks found under {ivy_root / 'blocks'}")

    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    uids: list[np.ndarray] = []
    model_keys: set[str] = set()

    for fp in tqdm(block_files, desc=f"load Ivy {expected_foundation}", leave=True):
        with np.load(fp, allow_pickle=False) as d:
            required = {"X", "label_int", "patch_uid", "model_key"}
            missing = required - set(d.files)
            if missing:
                raise KeyError(f"{fp}: missing required arrays {sorted(missing)}")

            X = np.asarray(d["X"], dtype=np.float32)
            y = np.asarray(d["label_int"], dtype=np.int64)
            uid = np.asarray(d["patch_uid"]).astype(str)
            model_keys.add(str(np.asarray(d["model_key"]).item()))

        if X.ndim != 2 or X.shape[1] != expected_dim:
            raise ValueError(
                f"{fp}: expected X=[n,{expected_dim}] for {expected_foundation}, got {X.shape}."
            )
        if not (len(X) == len(y) == len(uid)):
            raise ValueError(
                f"{fp}: X/label_int/patch_uid lengths disagree: {len(X)}, {len(y)}, {len(uid)}."
            )
        if not np.isfinite(X).all():
            raise ValueError(f"{fp}: non-finite Ivy embeddings detected.")
        unsupported = sorted(set(np.unique(y).tolist()) - set(IVY_SUPPORTED_CLASS_IDS))
        if unsupported:
            raise ValueError(
                f"{fp}: unexpected Ivy labels {unsupported}; expected only {list(IVY_SUPPORTED_CLASS_IDS)}."
            )

        Xs.append(X)
        ys.append(y)
        uids.append(uid)

    if len(model_keys) != 1:
        raise RuntimeError(
            f"{ivy_root}: found multiple model_key values across blocks: {sorted(model_keys)}"
        )
    expected_key = str(cfg.get("model_key", ""))
    if expected_key and next(iter(model_keys)) != expected_key:
        raise RuntimeError(
            f"{ivy_root}: block model_key disagrees with embedding_config.json."
        )

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    uid_all = np.concatenate(uids, axis=0)

    order = np.argsort(uid_all, kind="stable")
    X_all, y_all, uid_all = X_all[order], y_all[order], uid_all[order]
    repeated = uid_all[1:] == uid_all[:-1]
    if repeated.any():
        preview = uid_all[1:][repeated][:5].tolist()
        raise RuntimeError(
            f"{ivy_root}: found {int(repeated.sum()):,} duplicate patch_uid value(s), e.g. {preview}. "
            "Resolve the duplicate block state before training."
        )

    return X_all, y_all, uid_all, cfg


def _select_independent_ivy_rows(
    X: np.ndarray,
    y: np.ndarray,
    source_name: str,
    max_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Sample one source's Ivy pool without looking at any other source."""
    max_per_class = int(max_per_class)
    if max_per_class < 0:
        raise ValueError(
            "SWEEP.ivy_max_per_class must be >= 0; 0 means all rows per class."
        )

    source_offset = {"virchow2": 0, "hoptimus1": 10_000}.get(source_name, 20_000)
    selected_parts: list[np.ndarray] = []
    available_by_class: Dict[int, int] = {}
    selected_by_class: Dict[int, int] = {}

    for class_id in IVY_SUPPORTED_CLASS_IDS:
        idx = np.flatnonzero(y == class_id).astype(np.int64, copy=False)
        available_by_class[int(class_id)] = int(len(idx))
        if len(idx) == 0:
            raise RuntimeError(
                f"{source_name}: Ivy pool has no rows for class {class_id}."
            )
        if max_per_class > 0:
            if len(idx) < max_per_class:
                raise RuntimeError(
                    f"{source_name}: only {len(idx):,} Ivy rows available for class {class_id}, "
                    f"but ivy_max_per_class={max_per_class:,} was requested. "
                    "Use 0 for all rows or lower the requested cap."
                )
            rng = np.random.default_rng(int(seed) + source_offset + int(class_id))
            idx = rng.choice(idx, size=max_per_class, replace=False).astype(
                np.int64, copy=False
            )
        selected_parts.append(idx)
        selected_by_class[int(class_id)] = int(len(idx))

    selected = np.concatenate(selected_parts).astype(np.int64, copy=False)
    rng = np.random.default_rng(int(seed) + source_offset)
    selected = selected[rng.permutation(len(selected))]

    return (
        X[selected].astype(np.float32, copy=False),
        y[selected].astype(np.int64, copy=False),
        {
            "selection_seed": int(seed) + source_offset,
            "available_by_class": available_by_class,
            "selected_by_class": selected_by_class,
            "n_selected": int(len(selected)),
        },
    )


def _match_one_ivy_source_to_brats(
    X_brats: np.ndarray,
    X_ivy: np.ndarray,
    source_name: str,
    mode: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Match one raw Ivy source to the existing BraTS source representation."""
    mode = str(mode)
    if mode not in {"auto", "l2", "none"}:
        raise ValueError("SWEEP.ivy_norm_mode must be one of: auto, l2, none.")
    X_brats = np.asarray(X_brats, dtype=np.float32)
    X_ivy = np.asarray(X_ivy, dtype=np.float32)
    if X_brats.ndim != 2 or X_ivy.ndim != 2 or X_brats.shape[1] != X_ivy.shape[1]:
        raise ValueError(
            f"{source_name}: BraTS/Ivy dimension mismatch: {X_brats.shape} vs {X_ivy.shape}."
        )

    brats_stats = _norm_summary(X_brats)
    ivy_stats_before = _norm_summary(X_ivy)
    do_l2 = (mode == "l2") or (mode == "auto" and _looks_unit_normed(brats_stats))
    Xi = _row_l2_normalize(X_ivy) if do_l2 else X_ivy
    return Xi.astype(np.float32, copy=False), {
        "brats_norms": brats_stats,
        "ivy_norms_before": ivy_stats_before,
        "action": "row_l2_normalize_ivy_in_memory" if do_l2 else "leave_ivy_raw",
        "ivy_norms_after": _norm_summary(Xi),
    }


def load_independent_ivy_source_extras(
    X_brats: Mapping[str, np.ndarray],
    source_names: Sequence[str] | None = None,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    """Return independent Ivy extras for the configured sources present in X_brats.

    There is intentionally no UID intersection and no patient split for Ivy. Each
    source is sampled independently and later appended only to every training
    fold.
    """
    requested = tuple(source_names or tuple(X_brats.keys()))
    supported = tuple(s for s in requested if s in IVY_SOURCE_ROOT_ATTRS)
    if not supported:
        raise RuntimeError(
            f"Ivy augmentation requested, but none of the active sources have Ivy roots configured: {requested}."
        )
    missing = [s for s in supported if s not in X_brats]
    if missing:
        raise RuntimeError(
            f"BraTS source bundle lacks requested Ivy sources: {missing}."
        )

    extras: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    source_summary: Dict[str, Any] = {}
    for source_name in supported:
        root = _ivy_root_for_source(source_name)
        X_all, y_all, uid_all, cfg = _load_ivy_foundation_blocks(root, source_name)
        X_selected, y_selected, selection = _select_independent_ivy_rows(
            X_all,
            y_all,
            source_name=source_name,
            max_per_class=SWEEP.ivy_max_per_class,
            seed=SWEEP.ivy_seed,
        )
        X_matched, representation = _match_one_ivy_source_to_brats(
            X_brats[source_name],
            X_selected,
            source_name=source_name,
            mode=SWEEP.ivy_norm_mode,
        )
        extras[source_name] = (X_matched, y_selected)
        source_summary[source_name] = {
            "root": str(root.expanduser().resolve()),
            "model_key": str(cfg.get("model_key", "")),
            "embedding_dim": int(cfg.get("embedding_dim", X_all.shape[1])),
            "n_unique_rows_before_selection": int(len(uid_all)),
            "selection": selection,
            "representation_matching": representation,
        }

    return extras, {
        "augmentation_mode": "independent_per_source_train_only_no_uid_intersection",
        "ivy_max_per_class": int(SWEEP.ivy_max_per_class),
        "ivy_seed": int(SWEEP.ivy_seed),
        "ivy_norm_mode": str(SWEEP.ivy_norm_mode),
        "sources": source_summary,
    }


def _ivy_fingerprint_for_sources(source_names: Sequence[str]) -> str:
    """Fingerprint the Ivy train-only augmentation inputs used for config IDs."""
    supported = tuple(s for s in source_names if s in IVY_SOURCE_ROOT_ATTRS)
    if not supported:
        raise RuntimeError(
            f"Ivy augmentation requested, but none of these sources have Ivy roots configured: {tuple(source_names)}."
        )
    h = hashlib.sha1()
    h.update(b"ivy-train-only-v1\0")
    h.update(
        json.dumps(
            {
                "ivy_max_per_class": int(SWEEP.ivy_max_per_class),
                "ivy_seed": int(SWEEP.ivy_seed),
                "ivy_norm_mode": str(SWEEP.ivy_norm_mode),
                "supported_sources": supported,
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    h.update(b"\0")
    for source_name in supported:
        root = _ivy_root_for_source(source_name).expanduser().resolve()
        h.update(source_name.encode("utf-8"))
        h.update(b"\0")
        h.update(str(root).encode("utf-8"))
        h.update(b"\0")
        cfg_path = root / "embedding_config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing Ivy embedding configuration: {cfg_path}")
        h.update(cfg_path.read_bytes())
        h.update(b"\0")
        block_files = sorted((root / "blocks").glob("block-*.npz"))
        if not block_files:
            raise RuntimeError(f"No Ivy blocks found under {root / 'blocks'}")
        for fp in block_files:
            st = fp.stat()
            h.update(fp.name.encode("utf-8"))
            h.update(b"\0")
            h.update(str(int(st.st_size)).encode("ascii"))
            h.update(b"\0")
            h.update(str(int(st.st_mtime_ns)).encode("ascii"))
            h.update(b"\0")
    return h.hexdigest()


def train_aug_fingerprint_for(
    suffixes: Tuple[str, ...], source_names: Sequence[str]
) -> str:
    if not has_ivy_aug(suffixes):
        return ""
    return _ivy_fingerprint_for_sources(source_names)


def _extras_for_single_array(
    source_name: str,
    X: np.ndarray,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    extras, summary = load_independent_ivy_source_extras(
        {source_name: X}, source_names=(source_name,)
    )
    return extras, summary


# -----------------------------------------------------------------------------
# SQLite registry (shared across all configs) + per-config OOF npz "database"
# -----------------------------------------------------------------------------

_DB_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS config_results (
    config_id                      TEXT PRIMARY KEY,
    config_name                    TEXT NOT NULL,
    status                         TEXT NOT NULL CHECK(status IN ('completed', 'failed')),
    n_folds_completed              INTEGER NOT NULL,
    mcc_mean                       REAL,
    mcc_std                        REAL,
    recall_per_class_average_mean  REAL,
    recall_per_class_average_std   REAL,
    f1_per_class_average_mean      REAL,
    f1_per_class_average_std       REAL,
    specificity_per_class_weighted_mean REAL,
    specificity_per_class_weighted_std  REAL,
    auroc_per_class_weighted_mean  REAL,
    auroc_per_class_weighted_std   REAL,
    accuracy_global_mean           REAL,
    accuracy_global_std            REAL,
    f1_per_class_mean_json         TEXT,
    f1_per_class_std_json          TEXT,
    oof_path                       TEXT
);
"""


def db_connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_DB_SCHEMA)
    conn.commit()
    return conn


def terminal_config_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT config_id FROM config_results WHERE status='completed'"
    ).fetchall()
    return {str(row["config_id"]) for row in rows}


def get_result_status(conn: sqlite3.Connection, cid: str) -> str | None:
    row = conn.execute(
        "SELECT status FROM config_results WHERE config_id=?", (cid,)
    ).fetchone()
    return None if row is None else str(row["status"])


def upsert_result(
    conn: sqlite3.Connection,
    cid: str,
    name: str,
    status: str,
    n_folds: int,
    summary: Mapping[str, float],
    f1_class_mean: np.ndarray | None,
    f1_class_std: np.ndarray | None,
    oof_path: str | None,
) -> None:
    payload: Dict[str, Any] = {
        "config_id": cid,
        "config_name": name,
        "status": status,
        "n_folds_completed": int(n_folds),
        "f1_per_class_mean_json": json.dumps(
            [round(float(v), 6) for v in f1_class_mean]
        )
        if f1_class_mean is not None
        else None,
        "f1_per_class_std_json": json.dumps([round(float(v), 6) for v in f1_class_std])
        if f1_class_std is not None
        else None,
        "oof_path": oof_path,
    }
    payload.update({col: summary.get(col) for col in METRIC_SUMMARY_COLUMNS})
    columns = [
        "config_id",
        "config_name",
        "status",
        "n_folds_completed",
        *METRIC_SUMMARY_COLUMNS,
        "f1_per_class_mean_json",
        "f1_per_class_std_json",
        "oof_path",
    ]
    values = ", ".join(f":{c}" for c in columns)
    updates = ", ".join(f"{c}=excluded.{c}" for c in columns if c != "config_id")
    conn.execute(
        f"INSERT INTO config_results ({', '.join(columns)}) VALUES ({values}) "
        f"ON CONFLICT(config_id) DO UPDATE SET {updates}",
        payload,
    )
    conn.commit()


def _ensure_patch_index_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS patient_index (
            patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient    TEXT UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS patch_index (
            patch_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            patch_key  TEXT UNIQUE NOT NULL,
            patient_id INTEGER NOT NULL REFERENCES patient_index(patient_id)
        );
        """
    )
    conn.commit()


def resolve_patch_and_patient_ids(
    conn: sqlite3.Connection, patients: np.ndarray, names: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Map every (patient, name) row to a stable global int64 patch_id and
    int64 patient_id, backed by two tiny SQLite tables. This is the ONE place
    a string ("patient:name") ever gets built or compared in the whole
    pipeline. Every downstream artifact -- the OOF npz files, and every
    alignment step in ensembling_sweep.py -- works on these integers only.

    The tables only grow when a genuinely new patch_key is seen, which in
    practice happens once (the very first config saved this run); every
    later config, and every ensembling read, is a pure vectorized lookup.
    All row->id resolution below is vectorized (pandas hashtable-backed
    isin/reindex), never a per-row Python loop, so this stays fast even at
    millions of rows.
    """
    _ensure_patch_index_schema(conn)
    patch_key = np.array(
        [f"{p}:{n}" for p, n in zip(patients.tolist(), names.tolist())], dtype=object
    )

    # --- patients: tiny cardinality (a handful of rows), trivial either way ---
    existing_patients = pd.read_sql_query(
        "SELECT patient, patient_id FROM patient_index", conn
    )
    known_patients = (
        set(existing_patients["patient"]) if not existing_patients.empty else set()
    )
    new_patients = sorted(set(patients.tolist()) - known_patients)
    if new_patients:
        conn.executemany(
            "INSERT OR IGNORE INTO patient_index (patient) VALUES (?)",
            [(p,) for p in new_patients],
        )
        conn.commit()
        existing_patients = pd.read_sql_query(
            "SELECT patient, patient_id FROM patient_index", conn
        )
    patient_map = pd.Series(
        existing_patients["patient_id"].to_numpy(),
        index=existing_patients["patient"].to_numpy(),
    )
    patient_id = patient_map.reindex(patients).to_numpy(dtype=np.int64)

    # --- patches: up to millions of rows; every step below is vectorized ---
    existing_patches = pd.read_sql_query(
        "SELECT patch_key, patch_id FROM patch_index", conn
    )
    existing_index = (
        pd.Index(existing_patches["patch_key"])
        if not existing_patches.empty
        else pd.Index([], dtype=object)
    )
    is_new = ~pd.Index(patch_key).isin(
        existing_index
    )  # vectorized hashtable membership, not a python loop
    if is_new.any():
        first_seen = (
            pd.DataFrame({"patch_key": patch_key, "patient_id": patient_id})
            .drop_duplicates("patch_key")
            .set_index("patch_key")
        )
        new_keys = pd.unique(patch_key[is_new])
        rows_to_insert = list(
            zip(new_keys.tolist(), first_seen.loc[new_keys, "patient_id"].tolist())
        )
        conn.executemany(
            "INSERT OR IGNORE INTO patch_index (patch_key, patient_id) VALUES (?, ?)",
            rows_to_insert,
        )
        conn.commit()
        existing_patches = pd.read_sql_query(
            "SELECT patch_key, patch_id FROM patch_index", conn
        )

    patch_map = pd.Series(
        existing_patches["patch_id"].to_numpy(),
        index=existing_patches["patch_key"].to_numpy(),
    )
    patch_id = patch_map.reindex(patch_key).to_numpy(dtype=np.int64)
    return patch_id, patient_id


def save_oof(
    oof_dir: Path,
    cid: str,
    name: str,
    proba: np.ndarray,
    y: np.ndarray,
    fold_id: np.ndarray,
    patch_id: np.ndarray,
    patient_id: np.ndarray,
) -> str:
    """One OOF 'database' per config: an npz keyed by config_id, holding only
    validation rows (fold_id >= 0). Rows are sorted by patch_id ascending
    before saving -- this is what makes ensembling_sweep.py's alignment a
    single cheap numpy call: two configs covering the exact same patch set
    are then byte-identical in row order automatically, and two configs
    covering different subsets can still be intersected with one vectorized
    np.intersect1d call, no branching needed either way.

    Every column here is an integer or float array; no strings are stored,
    so nothing downstream ever pays for string hashing/comparison."""
    m = fold_id >= 0
    idx = np.flatnonzero(m)
    order = np.argsort(patch_id[idx], kind="stable")
    idx = idx[order]

    out_path = oof_dir / f"{cid}.npz"
    np.savez_compressed(
        out_path,
        proba=proba[idx].astype(np.float32),
        y=y[idx].astype(np.int64),
        fold_id=fold_id[idx].astype(np.int64),
        patch_id=patch_id[idx].astype(np.int64),
        patient_id=patient_id[idx].astype(np.int64),
        name=name,
    )
    return str(out_path)


# -----------------------------------------------------------------------------
# Runner: one config -> all folds, full OOF, no pruning
# -----------------------------------------------------------------------------


def run_one_config(
    conn: sqlite3.Connection,
    oof_dir: Path,
    cid: str,
    name: str,
    transform: str,
    estimator: Any,
    X_bundle: Any,
    y: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    patch_id: np.ndarray,
    patient_id: np.ndarray,
    overwrite: bool,
    source_train_extras: Mapping[str, Tuple[np.ndarray, np.ndarray]]
    | Tuple[np.ndarray, np.ndarray]
    | None = None,
    train_aug_summary: Mapping[str, Any] | None = None,
) -> str | None:
    existing_status = get_result_status(conn, cid)
    if existing_status is not None and not overwrite:
        print(f"[skip] {cid} {name}: status={existing_status}")
        return existing_status

    print(f"\n=== {cid} {name} ===")
    t0 = time.time()
    fold_metrics: List[Dict[str, float]] = []
    per_class_list: List[np.ndarray] = []

    oof_proba = np.full((len(y), N_CLASSES), np.nan, np.float32)
    oof_fold = np.full(len(y), -1, np.int64)

    try:
        for fold_idx, (tr, va) in enumerate(folds):
            t_fold = time.time()
            Xtr_raw = slice_bundle(X_bundle, tr)
            Xva_raw = slice_bundle(X_bundle, va)
            Xtr, Xva, fold_extras = transform_fold_with_extras(
                Xtr_raw, Xva_raw, transform, source_train_extras
            )
            model = clone(estimator)
            if fold_extras and _fit_accepts_source_train_extras(model):
                model.fit(Xtr, y[tr], source_train_extras=fold_extras)
            else:
                Xfit, yfit = append_train_extras_for_plain_estimator(
                    Xtr, y[tr], fold_extras
                )
                model.fit(Xfit, yfit)
                if fold_extras:
                    del Xfit, yfit
            proba = predict_proba_n(model, Xva, N_CLASSES, CLASS_IDS)

            metrics = compute_metrics(y[va], proba, N_CLASSES)
            pcf1 = per_class_f1(y[va], proba, N_CLASSES)
            train_support = np.bincount(y[tr], minlength=N_CLASSES)
            val_support = np.bincount(y[va], minlength=N_CLASSES)
            fold_metrics.append(metrics)
            per_class_list.append(pcf1)

            oof_proba[va] = proba
            oof_fold[va] = fold_idx

            pcf1_str = " ".join(f"c{c}={v:.3f}" for c, v in enumerate(pcf1))
            support_str = " ".join(
                f"c{c}={int(train_support[c])}/{int(val_support[c])}"
                for c in range(N_CLASSES)
            )
            print(
                f"fold {fold_idx}: mcc={metrics['mcc']:.4f} recall={metrics['recall_per_class_average']:.4f} "
                f"f1={metrics['f1_per_class_average']:.4f} spec={metrics['specificity_per_class_weighted']:.4f} "
                f"auroc={metrics['auroc_per_class_weighted']:.4f} acc={metrics['accuracy_global']:.4f} "
                f"({time.time() - t_fold:.1f}s)\n  per-class F1: {pcf1_str}"
                f"\n  support (train/val):    {support_str}"
            )
            del Xtr, Xva, model, proba
            gc.collect()

        summary: Dict[str, float] = {}
        for metric in METRIC_NAMES:
            values = np.asarray(
                [float(row[metric]) for row in fold_metrics], dtype=np.float64
            )
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=0))

        pc_stack = np.stack(per_class_list, axis=0)
        f1_class_mean = pc_stack.mean(axis=0)
        f1_class_std = pc_stack.std(axis=0, ddof=0)

        oof_path = save_oof(
            oof_dir, cid, name, oof_proba, y, oof_fold, patch_id, patient_id
        )
        upsert_result(
            conn,
            cid,
            name,
            "completed",
            len(fold_metrics),
            summary,
            f1_class_mean,
            f1_class_std,
            oof_path,
        )
        if train_aug_summary:
            atomic_json(
                oof_dir / f"{cid}_train_augmentation_summary.json",
                dict(train_aug_summary),
            )
        print(
            f"[stored] {cid}: status=completed; oof={oof_path}; elapsed={time.time() - t0:.1f}s"
        )
        return "completed"
    except Exception as exc:
        print(f"[failed] {cid} {name}: {exc}")
        return None


class ShrinkageMahalanobisProbe(BaseEstimator, ClassifierMixin):
    """Generative alternative to LinearProbe: per-class Gaussian means +
    a single shrinkage-regularized shared covariance (regularized LDA),
    classified by Mahalanobis distance -> softmax posterior, then the same
    per-class threshold tuning as LinearProbe. No SGD, no iterative fitting
    at all -- everything is closed-form (means + one covariance solve),
    so it's also a useful sanity-check baseline against the linear-SGD family.
    """

    def __init__(self, shrinkage=0.1, cal_frac=0.10, grid=15, passes=2, seed=42):
        self.shrinkage = shrinkage
        self.cal_frac = cal_frac
        self.grid = grid
        self.passes = passes
        self.seed = seed

    def _stack(self, X):
        def l2(a):
            a = np.asarray(a, np.float32)
            return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)

        if isinstance(X, dict):
            return np.concatenate([l2(X[k]) for k in sorted(X)], axis=1)
        return l2(X)

    def fit(self, X, y):
        y = np.asarray(y, np.int64)
        Z = self._stack(X)
        self.classes_ = CLASS_IDS.copy()

        fit_i, cal_i = next(
            StratifiedShuffleSplit(
                n_splits=1, test_size=self.cal_frac, random_state=self.seed
            ).split(Z, y)
        )
        Zf, yf = Z[fit_i], y[fit_i]

        d = Zf.shape[1]
        self.means_ = np.zeros((N_CLASSES, d), dtype=np.float64)
        pooled = np.zeros((d, d), dtype=np.float64)
        n_total = 0
        self.log_prior_ = np.full(N_CLASSES, -np.inf, dtype=np.float64)
        for c in range(N_CLASSES):
            Xc = Zf[yf == c]
            if len(Xc) < 2:
                continue
            mu = Xc.mean(axis=0)
            self.means_[c] = mu
            Xc_c = Xc - mu
            pooled += Xc_c.T @ Xc_c
            n_total += len(Xc)
            self.log_prior_[c] = np.log(len(Xc))
        self.log_prior_ -= np.nanmax(self.log_prior_[np.isfinite(self.log_prior_)])

        cov = pooled / max(n_total - 1, 1)
        trace_avg = np.trace(cov) / d
        cov_shrunk = (1 - self.shrinkage) * cov + self.shrinkage * trace_avg * np.eye(d)
        self.precision_ = np.linalg.pinv(cov_shrunk)

        self.thr_ = self._tune(y[cal_i], self._raw_proba(Z[cal_i]))
        return self

    def _raw_proba(self, Z):
        # class score = -0.5 * mahalanobis^2 + log_prior  (shared-cov LDA is
        # equivalent to a linear score in whitened space, but we keep the
        # explicit quadratic form so per-class covariance could later be
        # swapped in without changing the interface)
        scores = np.zeros((Z.shape[0], N_CLASSES), dtype=np.float64)
        for c in range(N_CLASSES):
            if not np.isfinite(self.log_prior_[c]):
                scores[:, c] = -1e9
                continue
            diff = Z - self.means_[c]
            maha2 = np.einsum("ij,jk,ik->i", diff, self.precision_, diff)
            scores[:, c] = -0.5 * maha2 + self.log_prior_[c]
        scores -= scores.max(axis=1, keepdims=True)
        e = np.exp(scores)
        return (e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)).astype(
            np.float32
        )

    def _tune(self, yc, pc):
        thr = np.ones(N_CLASSES, np.float32)
        grid = np.linspace(0.05, 2.0, int(self.grid), dtype=np.float32)
        labels = np.arange(N_CLASSES)
        score = lambda t: f1_score(
            yc,
            (pc / t.reshape(1, -1)).argmax(1),
            average="macro",
            zero_division=0,
            labels=labels,
        )
        best = score(thr)
        for _ in range(int(self.passes)):
            moved = False
            for c in range(N_CLASSES):
                best_c = thr[c]
                for t in grid:
                    cand = thr.copy()
                    cand[c] = t
                    s = score(cand)
                    if s > best + 1e-12:
                        best, best_c, moved = s, float(t), True
                thr[c] = best_c
            if not moved:
                break
        return thr.astype(np.float32)

    def predict_proba(self, X):
        p = self._raw_proba(self._stack(X))
        p = p / self.thr_.reshape(1, -1)
        return (p / np.clip(p.sum(1, keepdims=True), 1e-12, None)).astype(np.float32)

    def predict(self, X):
        return self.predict_proba(X).argmax(1).astype(np.int64)


# -----------------------------------------------------------------------------
# Ranking / printing
# -----------------------------------------------------------------------------

RANK_MEAN_METRICS: Tuple[str, ...] = (
    "f1_per_class_average",
    "mcc",
    "recall_per_class_average",
)
RANK_MEAN_COLUMN = "f1_mcc_recall_mean"


def load_result_table(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM config_results", conn)


def print_ranking(results: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    if results.empty:
        print("No completed configurations.")
        return results
    ranked = results.copy()
    ranked[RANK_MEAN_COLUMN] = ranked[[f"{m}_mean" for m in RANK_MEAN_METRICS]].mean(
        axis=1, skipna=False
    )
    sort_cols = [RANK_MEAN_COLUMN, *[f"{m}_mean" for m in RANK_COLUMNS]]
    ranked = ranked.sort_values(
        sort_cols, ascending=[False] * len(sort_cols), na_position="last"
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    view = ranked.rename(
        columns={
            "n_folds_completed": "folds",
            "f1_per_class_average_mean": "f1",
            "mcc_mean": "mcc",
            "recall_per_class_average_mean": "recall",
            RANK_MEAN_COLUMN: "mean(f1,mcc,recall)",
        }
    )[
        [
            "rank",
            "config_name",
            "status",
            "folds",
            "f1",
            "mcc",
            "recall",
            "mean(f1,mcc,recall)",
        ]
    ]
    print("\nCompleted configurations, ranked by equal-weight mean(f1, mcc, recall):")
    print(view.head(n).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return ranked


def print_per_class_f1(ranked: pd.DataFrame, n: int = 50) -> None:
    if ranked.empty:
        return
    rows = []
    for _, r in ranked.head(n).iterrows():
        try:
            vals = json.loads(r["f1_per_class_mean_json"])
        except Exception:
            continue
        row = {"rank": r["rank"], "config_name": r["config_name"]}
        row.update({f"c{c}": vals[c] for c in range(len(vals))})
        rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows)
    print(
        f"\nPer-class F1 (mean over folds). Note: a class being present in a "
        f"FORCE_TRAIN_PATIENTS patient does NOT mean it reads 0 here -- it only "
        f"does if that patient is also that class's only source. Run "
        f"diagnose_class_support.py to check any class's actual per-fold "
        f"validation coverage before assuming a low/zero score is structural "
        f"rather than a real model failure."
    )
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


class BalancedFitClassifier(BaseEstimator, ClassifierMixin):
    """Apply inverse-frequency class balancing to estimators supporting sample_weight."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        sw = compute_sample_weight("balanced", y)

        self.estimator_.fit(
            X,
            y,
            sample_weight=sw,
        )

        self.classes_ = getattr(
            self.estimator_,
            "classes_",
            np.unique(y),
        )
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


# =============================================================================
# EDIT HERE -- top-ranked configs from the search sweep, ported for OOF export.
# =============================================================================
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, (float, np.floating)):
        value = float(value)

        if np.isnan(value):
            return "__NaN__"
        if np.isposinf(value):
            return "__Infinity__"
        if np.isneginf(value):
            return "__-Infinity__"

        return value

    if isinstance(value, np.generic):
        return _canonicalize(value.item())

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(k): _canonicalize(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }

    if isinstance(value, (tuple, list)):
        return [_canonicalize(v) for v in value]

    if isinstance(value, np.ndarray):
        return _canonicalize(value.tolist())

    if hasattr(value, "get_params"):
        try:
            return {
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "params": _canonicalize(value.get_params(deep=False)),
            }
        except Exception:
            pass

    return {
        "class": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": str(value),
    }


class PermutedFeatureWrapper(BaseEstimator, ClassifierMixin):
    """Deterministically permute each source's feature dimensions, then fit
    the wrapped chunked ensemble. Contiguous chunks after permutation are
    therefore random non-overlapping feature partitions."""

    def __init__(self, base_estimator, random_state=42):
        self.base_estimator = base_estimator
        self.random_state = random_state

    def fit(self, X, y, source_train_extras=None):
        if not isinstance(X, Mapping):
            raise TypeError("Expected {foundation_name: array}.")

        self.permutations_ = {}

        for i, source_name in enumerate(sorted(X)):
            dim = int(np.asarray(X[source_name]).shape[1])
            rng = np.random.default_rng(int(self.random_state) + 1000 * (i + 1))
            self.permutations_[source_name] = rng.permutation(dim)

        X_perm = {k: np.asarray(v)[:, self.permutations_[k]] for k, v in X.items()}

        extras_perm = None
        if source_train_extras:
            extras_perm = {}
            for k, (Xe, ye) in source_train_extras.items():
                extras_perm[k] = (
                    np.asarray(Xe)[:, self.permutations_[k]],
                    np.asarray(ye),
                )

        self.base_ = clone(self.base_estimator)

        self.base_.fit(
            X_perm,
            y,
            source_train_extras=extras_perm,
        )

        self.classes_ = getattr(
            self.base_,
            "classes_",
            CLASS_IDS.copy(),
        )
        return self

    def _transform(self, X):
        return {k: np.asarray(v)[:, self.permutations_[k]] for k, v in X.items()}

    def predict_proba(self, X):
        return self.base_.predict_proba(self._transform(X))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1).astype(np.int64)


# ============================================================================
# Final-pipeline feature-representation ablation
#
# Fixed across all variants:
#   - V + H + G
#   - SGD + Ridge chunk heads
#   - Ivy train-only augmentation for V/H
#   - L2 normalization
#   - source-mean aggregation
#   - same rare-class policy / calibration
#
# Only feature representation changes.
# ============================================================================

FINAL_ABL_FOUNDATIONS = (
    "virchow2",
    "hoptimus1",
    "genbiopathfm",
)

FINAL_ABL_AUG = {
    "virchow2": ("ivy",),
    "hoptimus1": ("ivy",),
    "genbiopathfm": (),
}

FINAL_ABL_RIDGE_PROBES = {
    foundation: (
        (
            "ridge_lsqr_a10_l2_balanced",
            RidgeClassifier(
                alpha=10.0,
                class_weight="balanced",
                solver="lsqr",
                tol=1e-2,
                max_iter=50,
            ),
        ),
    )
    for foundation in FINAL_ABL_FOUNDATIONS
}

PROBE_CONFIGS: List[ProbeConfig] = [
    # ProbeConfig(
    #     name="mlp_h256_relu_a1e-4_lr1e-3_es_bal",
    #     estimator=BalancedFitClassifier(
    #         MLPClassifier(
    #             hidden_layer_sizes=(256,),
    #             activation="relu",
    #             solver="adam",
    #             alpha=1e-4,
    #             batch_size=256,
    #             learning_rate_init=1e-3,
    #             max_iter=50,
    #             early_stopping=True,
    #             validation_fraction=0.10,
    #             n_iter_no_change=5,
    #             random_state=SEED,
    #         )
    #     ),
    #     transform="l2",
    #     only_foundations=("virchow2",),
    #     aug_artifact_suffixes=(),
    # ),
    # ProbeConfig(
    #     name="xgb_hist_d4_n50_lr0.1_sub0.8_col0.5_bal",
    #     estimator=BalancedFitClassifier(
    #         XGBClassifier(
    #             objective="multi:softprob",
    #             num_class=N_CLASSES,
    #             n_estimators=50,
    #             max_depth=4,
    #             learning_rate=0.1,
    #             tree_method="hist",
    #             subsample=0.8,
    #             colsample_bytree=0.5,
    #             reg_lambda=1.0,
    #             random_state=SEED,
    #             n_jobs=-1,
    #             verbosity=0,
    #         )
    #     ),
    #     transform="l2",
    #     only_foundations=("virchow2",),
    #     aug_artifact_suffixes=(),
    # ),
    # ProbeConfig(
    #     name="logreg_multinomial_c1",
    #     estimator=LogisticRegression(
    #         C=1.0,
    #         penalty="l2",
    #         solver="lbfgs",
    #         class_weight="balanced",
    #         max_iter=1_000,
    #         random_state=SEED,
    #         n_jobs=1,
    #     ),
    #     transform="l2",
    #     only_foundations=("genbiopathfm",),
    # ),
    # ProbeConfig(
    #     "linear_balanced_thr_minf1-0.43",
    #     LinearProbe(alpha=3e-5, max_iter=25, grid=15, passes=2, seed=SEED),
    #     "none",
    #     only_foundations=("virchow2", "genbiopathfm"),
    # ),
    # ProbeConfig(
    #     "linear_balanced_thr_iter50",
    #     LinearProbe(alpha=3e-5, max_iter=50, grid=15, passes=2, seed=SEED),
    #     "none",
    #     only_foundations=("virchow2",),
    # ),
    # ProbeConfig(
    #     "tta_mask_linear_n8_keep90",
    #     TTAProbe(
    #         base_estimator=LinearProbe(alpha=3e-5, max_iter=25, grid=15, passes=2, seed=SEED),
    #         n_aug=8, keep_prob=0.9, gaussian_std=0.0, include_clean=True, random_state=SEED,
    #     ),
    #     "none",
    #     only_foundations=("virchow2",),
    # ),
    # ProbeConfig(
    #     "sgd_logloss_a3e-5_l2",
    #     _sgd(alpha=3e-5, loss="log_loss", max_iter=20, seed=SEED),
    #     "l2",
    #     only_foundations=("provgigapath",),
    #     aug_artifact_suffixes=(),
    # ),
    # ProbeConfig(
    #     "ridge_lsqr_a10_l2_balanced",
    #     RidgeClassifier(alpha=10.0, class_weight="balanced", solver="lsqr", tol=1e-2, max_iter=50),
    #     "l2",
    #     only_foundations=("genbiopathfm",),
    #     aug_artifact_suffixes=(),
    # ),
    # ProbeConfig(
    #     "shrinkage_mahalanobis_s0.1",
    #     ShrinkageMahalanobisProbe(shrinkage=0.1, cal_frac=0.10, grid=15, passes=2, seed=SEED),
    #     "none",
    #     only_foundations=("hoptimus1",),
    # aug_artifact_suffixes=(),
    # ),
    ProbeConfig(
        "sgd_logloss_a3e-5_l2_test",
        _sgd(alpha=3e-5, loss="log_loss", max_iter=20, seed=SEED),
        "l2",
        only_foundations=("virchow2",),
    ),
    # ProbeConfig(
    #     "ridge_lsqr_a10_l2_balanced_13shard_local_only",
    #     RidgeClassifier(alpha=10.0, class_weight="balanced", solver="lsqr", tol=1e-2, max_iter=50),
    #     "l2",
    #     only_foundations=("virchow2",),
    #     aug_artifact_suffixes=("stainaug_local"),
    # ),
    # ProbeConfig(
    #     "softcentroid_cos_t007_s01_clean",
    #     SoftCentroidProbe(metric="cosine", temperature=0.07, shrinkage=0.10),
    #     "l2",
    #     only_foundations=("hoptimus1"),
    # ),
    # ProbeConfig(
    #     "ridge_lsqr_a10_l2_balanced_all",
    #     RidgeClassifier(alpha=10.0, class_weight="balanced", solver="lsqr", tol=1e-2, max_iter=50),
    #     "l2",
    #     only_foundations=("virchow2",),
    #     aug_artifact_suffixes=("stainaug", "stainaug_local"),
    # ),
    # ProbeConfig(
    #     "ridge_lsqr_a10_l2_balanced",
    #     RidgeClassifier(alpha=10.0, class_weight="balanced", solver="lsqr", tol=1e-2, max_iter=50),
    #     "l2",
    #     only_foundations=("virchow2", "hoptimus1"),
    #     aug_artifact_suffixes=("stainaug", "stainaug_local", "ivy"),
    # ),
    # ProbeConfig(
    #     "ridge_lsqr_a10_l2_balanced_ivy_cap8k",
    #     RidgeClassifier(alpha=10.0, class_weight="balanced", solver="lsqr", tol=1e-2, max_iter=50),
    #     "l2",
    #     only_foundations=("virchow2",),
    #     aug_artifact_suffixes=("ivy",),
    # ),
]

# ============================================================================
# Paste AFTER BratsPath2025ChunkedSGDEnsemble / PermutedFeatureWrapper support
# and BEFORE COMPOSITE_PROBE_CONFIGS.
# ============================================================================


class JointSGDRidgeChunkedEnsemble(BratsPath2025ChunkedSGDEnsemble):
    """
    Final-pipeline chunk ensemble:
      - primary SGD head on every source/chunk
      - additional Ridge head on the exact same source/chunk rows
      - SGD + Ridge heads are pooled BEFORE source aggregation/calibration
    """

    def __init__(
        self,
        base_estimator: Any | None = None,
        probe_estimators: Mapping[str, Sequence[Tuple[str, Any]]] | None = None,
        chunk_size: int = 768,
        min_chunk_dim: int = 16,
        rare_boost: float = 1.10,
        rare_quantile: float = 0.35,
        rare_classes: Tuple[int, ...] | None = None,
        calibration_fraction: float = 0.10,
        threshold_grid_size: int = 11,
        threshold_passes: int = 1,
        max_train_samples_per_class: int | None = None,
        use_sample_weight: bool = False,
        source_weights: Mapping[str, float] | None = None,
        source_aggregation: str = "source_mean",
        foundation_names: Sequence[str] | None = None,
        foundation_label: str | None = None,
        verbose: int = 0,
        random_state: int = SEED,
    ):
        super().__init__(
            base_estimator=base_estimator,
            chunk_size=chunk_size,
            min_chunk_dim=min_chunk_dim,
            rare_boost=rare_boost,
            rare_quantile=rare_quantile,
            rare_classes=rare_classes,
            calibration_fraction=calibration_fraction,
            threshold_grid_size=threshold_grid_size,
            threshold_passes=threshold_passes,
            max_train_samples_per_class=max_train_samples_per_class,
            use_sample_weight=use_sample_weight,
            source_weights=source_weights,
            source_aggregation=source_aggregation,
            foundation_names=foundation_names,
            foundation_label=foundation_label,
            verbose=verbose,
            random_state=random_state,
        )
        self.probe_estimators = probe_estimators

    def _fit_heads(
        self,
        X: Mapping[str, np.ndarray],
        y_by_source: Mapping[str, np.ndarray],
    ):
        # First fit the normal primary SGD heads.
        super()._fit_heads(X, y_by_source)

        heads = list(self.heads_)
        self.probe_heads_ = []

        probes_by_source = dict(self.probe_estimators or {})

        unknown = set(probes_by_source) - set(X)
        if unknown:
            raise KeyError(
                f"Probe estimators supplied for absent source(s): {sorted(unknown)}"
            )

        probe_jobs = []

        for source_name in sorted(probes_by_source):
            dim = int(np.asarray(X[source_name]).shape[1])

            for probe_name, estimator_prototype in probes_by_source[source_name]:
                for start, stop in self._iter_chunks(dim):
                    probe_jobs.append(
                        (
                            probe_name,
                            source_name,
                            int(start),
                            int(stop),
                            estimator_prototype,
                        )
                    )

        for i, (
            probe_name,
            source_name,
            start,
            stop,
            estimator_prototype,
        ) in enumerate(probe_jobs):
            Xchunk = np.asarray(
                X[source_name],
                dtype=np.float32,
            )[:, start:stop]

            y_source = np.asarray(
                y_by_source[source_name],
                dtype=np.int64,
            )

            sw = (
                compute_sample_weight("balanced", y_source)
                if bool(self.use_sample_weight)
                else None
            )

            clf = clone(estimator_prototype)

            params = getattr(clf, "get_params", lambda: {})()
            if "random_state" in params:
                clf.set_params(random_state=int(self.random_state) + 1_000_000 + i)

            if sw is None:
                clf.fit(Xchunk, y_source)
            else:
                try:
                    clf.fit(Xchunk, y_source, sample_weight=sw)
                except TypeError:
                    clf.fit(Xchunk, y_source)

            heads.append(
                (
                    source_name,
                    int(start),
                    int(stop),
                    clf,
                )
            )

            self.probe_heads_.append(
                {
                    "probe_name": probe_name,
                    "source": source_name,
                    "start": int(start),
                    "stop": int(stop),
                }
            )

        # Calibration performed later by parent fit() now sees
        # the JOINT SGD + Ridge head pool.
        self.heads_ = heads
        return self


class PermutedFeatureWrapper(BaseEstimator, ClassifierMixin):
    """
    Deterministically permute each foundation's feature dimensions before
    fitting. Contiguous 768-d chunks after permutation therefore form
    random, non-overlapping, exhaustive 768-d partitions.

    The identical permutation is applied to BraTS rows, validation rows,
    and all train-only augmentation rows.
    """

    def __init__(
        self,
        base_estimator,
        random_state=SEED,
    ):
        self.base_estimator = base_estimator
        self.random_state = random_state

    def fit(
        self,
        X,
        y,
        source_train_extras=None,
    ):
        if not isinstance(X, Mapping):
            raise TypeError(
                "PermutedFeatureWrapper expects {foundation_name: array} input."
            )

        self.permutations_ = {}

        for i, source_name in enumerate(sorted(X)):
            dim = int(np.asarray(X[source_name]).shape[1])

            rng = np.random.default_rng(int(self.random_state) + 1000 * (i + 1))

            self.permutations_[source_name] = rng.permutation(dim)

        X_perm = {
            source_name: np.asarray(X_source)[:, self.permutations_[source_name]]
            for source_name, X_source in X.items()
        }

        extras_perm = None

        if source_train_extras:
            extras_perm = {}

            for source_name, (X_extra, y_extra) in source_train_extras.items():
                extras_perm[source_name] = (
                    np.asarray(X_extra)[:, self.permutations_[source_name]],
                    np.asarray(y_extra),
                )

        self.base_ = clone(self.base_estimator)

        self.base_.fit(
            X_perm,
            y,
            source_train_extras=extras_perm,
        )

        self.classes_ = getattr(
            self.base_,
            "classes_",
            CLASS_IDS.copy(),
        )

        return self

    def _transform(self, X):
        return {
            source_name: np.asarray(X_source)[:, self.permutations_[source_name]]
            for source_name, X_source in X.items()
        }

    def predict_proba(self, X):
        return self.base_.predict_proba(self._transform(X))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1).astype(np.int64)


FINAL_ABL_FOUNDATIONS = (
    "virchow2",
    "hoptimus1",
    "genbiopathfm",
)

# Faithful final training augmentation for ALL THREE foundations.
FINAL_ABL_AUG = {
    "virchow2": (
        "ivy",
        "stainaug",
        "stainaug_local",
    ),
    "hoptimus1": (
        "ivy",
        "stainaug",
        "stainaug_local",
    ),
    "genbiopathfm": (
        "ivy",
        "stainaug",
        "stainaug_local",
    ),
}


def _final_ridge():
    return RidgeClassifier(
        alpha=10.0,
        class_weight="balanced",
        solver="lsqr",
        tol=1e-2,
        max_iter=50,
    )


# One Ridge head for every chunk of every foundation,
# alongside the primary SGD head.
FINAL_ABL_RIDGE_PROBES = {
    source_name: (
        (
            "ridge_lsqr_a10_l2_balanced",
            _final_ridge(),
        ),
    )
    for source_name in FINAL_ABL_FOUNDATIONS
}


COMPOSITE_PROBE_CONFIGS: List[CompositeProbeConfig] = [
    # ============================================================================
    # Paste these THREE entries inside COMPOSITE_PROBE_CONFIGS = [...]
    # ============================================================================
    # ---------------------------------------------------------------------------
    # 1. FULL EMBEDDING
    #    SGD + Ridge jointly; one full-dimensional head of each type per source.
    # ---------------------------------------------------------------------------
    # CompositeProbeConfig(
    #     name="finalabl_full_sgd_ridge",
    #     foundation_names=FINAL_ABL_FOUNDATIONS,
    #     estimator=JointSGDRidgeChunkedEnsemble(
    #         base_estimator=_sgd(
    #             alpha=3e-5,
    #             loss="log_loss",
    #             max_iter=20,
    #             seed=SEED,
    #         ),
    #         probe_estimators=FINAL_ABL_RIDGE_PROBES,
    #         chunk_size=1_000_000,
    #         min_chunk_dim=16,
    #         rare_boost=1.10,
    #         rare_quantile=0.35,
    #         rare_classes=None,
    #         calibration_fraction=0.10,
    #         threshold_grid_size=11,
    #         threshold_passes=1,
    #         max_train_samples_per_class=None,
    #         use_sample_weight=False,
    #         source_weights=None,
    #         source_aggregation="source_mean",
    #         random_state=SEED,
    #     ),
    #     transform="l2",
    #     per_foundation_aug_artifact_suffixes=FINAL_ABL_AUG,
    # ),
    # # ---------------------------------------------------------------------------
    # # 2. RANDOM NON-OVERLAPPING 768-D PARTITIONS
    # #    Same final pipeline; only feature ordering changes.
    # # ---------------------------------------------------------------------------
    # CompositeProbeConfig(
    #     name="finalabl_random768_s42_sgd_ridge",
    #     foundation_names=FINAL_ABL_FOUNDATIONS,
    #     estimator=PermutedFeatureWrapper(
    #         base_estimator=JointSGDRidgeChunkedEnsemble(
    #             base_estimator=_sgd(
    #                 alpha=3e-5,
    #                 loss="log_loss",
    #                 max_iter=20,
    #                 seed=SEED,
    #             ),
    #             probe_estimators=FINAL_ABL_RIDGE_PROBES,
    #             chunk_size=768,
    #             min_chunk_dim=16,
    #             rare_boost=1.10,
    #             rare_quantile=0.35,
    #             rare_classes=None,
    #             calibration_fraction=0.10,
    #             threshold_grid_size=11,
    #             threshold_passes=1,
    #             max_train_samples_per_class=None,
    #             use_sample_weight=False,
    #             source_weights=None,
    #             source_aggregation="source_mean",
    #             random_state=SEED,
    #         ),
    #         random_state=SEED,
    #     ),
    #     transform="l2",
    #     per_foundation_aug_artifact_suffixes=FINAL_ABL_AUG,
    # ),
    # # ---------------------------------------------------------------------------
    # # 3. CONTIGUOUS 768-D BLOCKS
    # #    Current final representation.
    # # ---------------------------------------------------------------------------
    # CompositeProbeConfig(
    #     name="finalabl_contig768_sgd_ridge",
    #     foundation_names=FINAL_ABL_FOUNDATIONS,
    #     estimator=JointSGDRidgeChunkedEnsemble(
    #         base_estimator=_sgd(
    #             alpha=3e-5,
    #             loss="log_loss",
    #             max_iter=20,
    #             seed=SEED,
    #         ),
    #         probe_estimators=FINAL_ABL_RIDGE_PROBES,
    #         chunk_size=768,
    #         min_chunk_dim=16,
    #         rare_boost=1.10,
    #         rare_quantile=0.35,
    #         rare_classes=None,
    #         calibration_fraction=0.10,
    #         threshold_grid_size=11,
    #         threshold_passes=1,
    #         max_train_samples_per_class=None,
    #         use_sample_weight=False,
    #         source_weights=None,
    #         source_aggregation="source_mean",
    #         random_state=SEED,
    #     ),
    #     transform="l2",
    #     per_foundation_aug_artifact_suffixes=FINAL_ABL_AUG,
    # ),
]
#     # To run a composite over augmented rows, either apply one suffix tuple to every
#     # source with aug_artifact_suffixes=("stainaug",), or override per foundation
#     # with per_foundation_aug_artifact_suffixes={"hoptimus1": ("stainaug",), "virchow2": ()}.
#     # Add "ivy" to either field for train-only Ivy GAP rows, e.g.
#     # aug_artifact_suffixes=("ivy",) or ("ivy", "stainaug", "stainaug_local").
#     # Ivy is never materialized into validation/OOF; it is appended to each fold's
#     # training fit through source_train_extras, matching the train/inference script.
#     # CompositeProbeConfig(
#     #     name="chunksgd_a3e-5_c1024_source",
#     #     foundation_names=("virchow2", "hoptimus1", "genbiopathfm"),
#     #     estimator=BratsPath2025ChunkedSGDEnsemble(
#     #         base_estimator=_sgd(alpha=3e-5, loss="log_loss", max_iter=20, seed=SEED),
#     #         chunk_size=1024, source_aggregation="source_mean", random_state=SEED,
#     #     ),
#     #     transform="none",
#     # ),
#     # CompositeProbeConfig(
#     #     name="chunksgd_a3e-5_c1024_source",
#     #     foundation_names=("virchow2", "genbiopathfm"),
#     #     estimator=BratsPath2025ChunkedSGDEnsemble(
#     #         base_estimator=_sgd(alpha=3e-5, loss="log_loss", max_iter=20, seed=SEED),
#     #         chunk_size=1024, source_aggregation="source_mean", random_state=SEED,
#     #     ),
#     #     transform="none",
#     # ),
#     # CompositeProbeConfig(
#     #     name="chunksgd_a3e-5_c512_source",
#     #     foundation_names=("virchow2", "genbiopathfm"),
#     #     estimator=BratsPath2025ChunkedSGDEnsemble(
#     #         base_estimator=_sgd(alpha=3e-5, loss="log_loss", max_iter=20, seed=SEED),
#     #         chunk_size=512, source_aggregation="source_mean", random_state=SEED,
#     #     ),
#     #     transform="none",
#     # ),
#     # CompositeProbeConfig(
#     #     name="chunksgd_a3e-5_c1024",
#     #     foundation_names=("virchow2", "hoptimus1"),
#     #     estimator=BratsPath2025ChunkedSGDEnsemble(
#     #         base_estimator=_sgd(alpha=3e-5, loss="log_loss", max_iter=20, seed=SEED),
#     #         chunk_size=1024, source_aggregation="source_mean", random_state=SEED,
#     #     ),
#     #     transform="none",
#     #     aug_artifact_suffixes=("ivy", "stainaug_local"),
#     # ),
#     # CompositeProbeConfig(
#     #     name="chunksgd_a3e-5_c1024-cal005",
#     #     foundation_names=("virchow2", "hoptimus1"),
#     #     estimator=BratsPath2025ChunkedSGDEnsemble(
#     #         base_estimator=_sgd(alpha=3e-5, loss="log_loss", max_iter=20, seed=SEED),
#     #         chunk_size=1024, source_aggregation="source_mean", random_state=SEED,
#     #         threshold_grid_size=0.05,
#     #     ),
#     #     transform="none",
#     #     aug_artifact_suffixes=(),
#     # ),
#     # CompositeProbeConfig(
#     #     name="chunksgd_a3e-5_c1024_source",
#     #     foundation_names=("virchow2", "provgigapath", "genbiopathfm"),
#     #     estimator=BratsPath2025ChunkedSGDEnsemble(
#     #         base_estimator=_sgd(alpha=3e-5, loss="log_loss", max_iter=20, seed=SEED),
#     #         chunk_size=1024, source_aggregation="source_mean", random_state=SEED,
#     #     ),
#     #     transform="none",
#     # ),
#     # CompositeProbeConfig(
#     #     name="chunksgd_a3e-5_c1024",
#     #     foundation_names=("virchow2", "provgigapath"),
#     #     estimator=BratsPath2025ChunkedSGDEnsemble(
#     #         base_estimator=_sgd(alpha=3e-5, loss="log_loss", max_iter=20, seed=SEED),
#     #         chunk_size=1024, source_aggregation="head_mean", random_state=SEED,
#     #     ),
#     #     transform="none",
#     # ),
#     # CompositeProbeConfig(
#     #     name="chunksgd_a3e-5_c1024",
#     #     foundation_names=("virchow2",),
#     #     estimator=BratsPath2025ChunkedSGDEnsemble(
#     #         base_estimator=_sgd(alpha=3e-5, loss="log_loss", max_iter=20, seed=SEED),
#     #         chunk_size=1024, source_aggregation="source_mean", random_state=SEED,
#     #     ),
#     #     transform="none",
#     # ),
#     # CompositeProbeConfig(
#     #     name="featmask_a3e-5_m12_f0.5_source",
#     #     foundation_names=("hoptimus1", "virchow2"),
#     #     estimator=FeatureMaskedSGDEnsemble(
#     #         base_estimator=_sgd(alpha=3e-5, loss="log_loss", max_iter=20, seed=SEED),
#     #         n_masks_per_source=12, mask_fraction=0.5, min_mask_dim=16, mask_scheme="balanced",
#     #         source_aggregation="source_mean", random_state=SEED,
#     #     ),
#     #     transform="none",
#     #     aug_artifact_suffixes=("stainaug", "stainaug_local"),
#     # ),
#     # CompositeProbeConfig(
#     #     name="featmask_ridge_lsqr_a10_m4_f0.5_source",
#     #     foundation_names=("virchow2", "hoptimus1"),
#     #     estimator=FeatureMaskedSGDEnsemble(
#     #         base_estimator=RidgeClassifier(
#     #             alpha=10.0,
#     #             class_weight="balanced",
#     #             solver="lsqr",
#     #             tol=1e-2,
#     #             max_iter=50,
#     #         ),
#     #         n_masks_per_source=4,
#     #         mask_fraction=0.5,
#     #         min_mask_dim=16,
#     #         mask_scheme="balanced",
#     #         source_aggregation="source_mean",
#     #         random_state=SEED,
#     #     ),
#     #     transform="l2",
#     #     aug_artifact_suffixes=("stainaug",),
#     # ),
#     # CompositeProbeConfig(
#     #     name="chunksgd_a3e-5_c1024",
#     #     foundation_names=("virchow2", "hoptimus1"),
#     #     estimator=BratsPath2025ChunkedSGDEnsemble(
#     #         base_estimator=RidgeClassifier(
#     #             alpha=10.0,
#     #             class_weight="balanced",
#     #             solver="lsqr",
#     #             tol=1e-2,
#     #             max_iter=50,
#     #         ),            chunk_size=1024, source_aggregation="source_mean", random_state=SEED,
#     #     ),
#     #     transform="none",
#     #     aug_artifact_suffixes=("stainaug", "stainaug_local"),
#     # ),
#     # 1. Full embedding: one SGD head sees all Virchow2 dimensions
#     # CompositeProbeConfig(
#     #     name="featabl_full_sgd",
#     #     foundation_names=("virchow2",),
#     #     estimator=BratsPath2025ChunkedSGDEnsemble(
#     #         base_estimator=_sgd(
#     #             alpha=3e-5,
#     #             loss="log_loss",
#     #             max_iter=20,
#     #             seed=SEED,
#     #         ),
#     #         chunk_size=1_000_000,
#     #         source_aggregation="source_mean",
#     #         random_state=SEED,
#     #     ),
#     #     transform="l2",
#     #     aug_artifact_suffixes=(),
#     # ),

#     # # 2. Random non-overlapping 768-d partitions
#     # CompositeProbeConfig(
#     #     name="featabl_random768_s42_sgd",
#     #     foundation_names=("virchow2",),
#     #     estimator=PermutedFeatureWrapper(
#     #         base_estimator=BratsPath2025ChunkedSGDEnsemble(
#     #             base_estimator=_sgd(
#     #                 alpha=3e-5,
#     #                 loss="log_loss",
#     #                 max_iter=20,
#     #                 seed=SEED,
#     #             ),
#     #             chunk_size=768,
#     #             source_aggregation="source_mean",
#     #             random_state=SEED,
#     #         ),
#     #         random_state=SEED,
#     #     ),
#     #     transform="l2",
#     #     aug_artifact_suffixes=(),
#     # ),

#     # # 3. Contiguous 768-d blocks
#     # CompositeProbeConfig(
#     #     name="featabl_contig768_sgd",
#     #     foundation_names=("virchow2",),
#     #     estimator=BratsPath2025ChunkedSGDEnsemble(
#     #         base_estimator=_sgd(
#     #             alpha=3e-5,
#     #             loss="log_loss",
#     #             max_iter=20,
#     #             seed=SEED,
#     #         ),
#     #         chunk_size=768,
#     #         source_aggregation="source_mean",
#     #         random_state=SEED,
#     #     ),
#     #     transform="l2",
#     #     aug_artifact_suffixes=(),
#     # ),
#     CompositeProbeConfig(
#         name="finalabl_full_sgd_ridge",
#         foundation_names=FINAL_ABL_FOUNDATIONS,
#         estimator=BratsPath2025ChunkedSGDEnsemble(
#             base_estimator=_sgd(
#                 alpha=3e-5,
#                 loss="log_loss",
#                 max_iter=20,
#                 seed=SEED,
#             ),
#             chunk_size=1_000_000,
#             min_chunk_dim=16,

#             rare_boost=1.10,
#             rare_quantile=0.35,
#             rare_classes=None,

#             calibration_fraction=0.10,
#             threshold_grid_size=11,
#             threshold_passes=1,

#             max_train_samples_per_class=None,
#             use_sample_weight=False,

#             source_weights=None,
#             source_aggregation="source_mean",

#             probe_estimators=FINAL_ABL_RIDGE_PROBES,

#             random_state=SEED,
#         ),
#         transform="l2",
#         per_foundation_aug_artifact_suffixes=FINAL_ABL_AUG,
#     ),
# ]


def probe_runs_on_foundation(probe_config: ProbeConfig, foundation: str) -> bool:
    return (
        not probe_config.only_foundations or foundation in probe_config.only_foundations
    )


def main() -> None:
    out_dir = SWEEP.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_dir = out_dir / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)

    available = sorted(discover_available_foundations(SWEEP.embedding_root))
    print(f"foundations with embeddings on disk: {available}")
    active_foundations = [f for f in FOUNDATIONS if f in available]
    missing = sorted(set(FOUNDATIONS) - set(available))
    if missing:
        print(f"[warn] foundations listed but not found on disk, skipping: {missing}")
    print(f"base foundations selected for this run: {active_foundations}")

    conn = db_connect(out_dir / "configs.db")
    terminal_ids = terminal_config_ids(conn)
    print(f"terminal (completed) results already registered: {len(terminal_ids)}")
    print(
        "pruning: DISABLED -- every active config runs all folds for full OOF coverage."
    )

    merged_name_cache: Dict[Tuple[str, Tuple[str, ...]], str] = {}

    # -----------------------------
    # Single-foundation probe configs
    # -----------------------------
    for foundation in active_foundations:
        pending = [
            pc for pc in PROBE_CONFIGS if probe_runs_on_foundation(pc, foundation)
        ]
        if not pending:
            continue

        pending_by_target: Dict[
            str, List[Tuple[str, str, ProbeConfig, Tuple[str, ...]]]
        ] = {}
        for pc in pending:
            suffixes = tuple(pc.aug_artifact_suffixes)
            materialized_suffixes = non_ivy_aug_suffixes(suffixes)
            try:
                target_foundation = materialize_foundation(
                    foundation, materialized_suffixes, merged_name_cache
                )
                manifest = embedding_manifest_fingerprint(
                    SWEEP.embedding_root, (target_foundation,)
                )
                train_aug_fp = train_aug_fingerprint_for(suffixes, (foundation,))
            except Exception as exc:
                print(
                    f"[warn] single probe {pc.name!r} on {foundation!r} skipped: {exc}"
                )
                continue

            foundation_label = label_with_train_augmentations(
                target_foundation, suffixes
            )
            name = config_name(foundation_label, pc.name, pc.transform)
            cid = config_id(
                name, pc.estimator, (target_foundation,), manifest, train_aug_fp
            )
            if cid in terminal_ids and not SWEEP.overwrite:
                print(f"[skip] {cid} {name}: already registered")
            else:
                pending_by_target.setdefault(target_foundation, []).append(
                    (cid, name, pc, suffixes)
                )

        for target_foundation, pending_new in pending_by_target.items():
            if not pending_new:
                continue
            X, y, names, patients, slides, slide_groups = load_embeddings(
                SWEEP.embedding_root, target_foundation, SWEEP.split, SWEEP.max_files
            )
            print(
                f"{target_foundation}: X={X.shape}, y={y.shape}, "
                f"patients={len(np.unique(patients))}, slides={len(np.unique(slide_groups))}"
            )
            folds = build_folds(
                y,
                patients,
                slide_groups,
                GROUP_BY,
                N_SPLITS,
                SEED,
                force_train_patients=FORCE_TRAIN_PATIENTS,
            )
            patch_id, patient_id = resolve_patch_and_patient_ids(conn, patients, names)

            ivy_extras_by_source: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            ivy_summary: Dict[str, Any] | None = None
            if any(has_ivy_aug(suffixes) for _, _, _, suffixes in pending_new):
                print(
                    f"\nLoading train-only Ivy extras for single foundation {foundation!r}"
                )
                ivy_extras_by_source, ivy_summary = _extras_for_single_array(
                    foundation, X
                )
                n_ivy = (
                    int(len(next(iter(ivy_extras_by_source.values()))[1]))
                    if ivy_extras_by_source
                    else 0
                )
                print(f"  {foundation}: Ivy train-only extra rows = {n_ivy:,}")

            for cid, name, pc, suffixes in pending_new:
                use_ivy = has_ivy_aug(suffixes)
                status = run_one_config(
                    conn,
                    oof_dir,
                    cid,
                    name,
                    pc.transform,
                    pc.estimator,
                    X,
                    y,
                    folds,
                    patch_id,
                    patient_id,
                    SWEEP.overwrite,
                    source_train_extras=ivy_extras_by_source if use_ivy else None,
                    train_aug_summary=ivy_summary if use_ivy else None,
                )
                if status == "completed":
                    terminal_ids.add(cid)

            del X, y, names, patients, slides, slide_groups
            gc.collect()

    # -----------------------------
    # Multi-foundation composite configs
    # -----------------------------
    for cc in COMPOSITE_PROBE_CONFIGS:
        base_missing = [fn for fn in cc.foundation_names if fn not in available]
        if base_missing:
            print(
                f"[warn] composite probe {cc.name!r} needs missing base foundations {base_missing}, skipping."
            )
            continue

        per_source_suffixes = {
            fn: composite_suffixes_for(cc, fn) for fn in cc.foundation_names
        }
        all_suffixes = tuple(
            s for fn in cc.foundation_names for s in per_source_suffixes[fn]
        )
        ivy_source_names = tuple(
            fn for fn in cc.foundation_names if has_ivy_aug(per_source_suffixes[fn])
        )
        try:
            target_foundations = tuple(
                materialize_foundation(
                    fn, non_ivy_aug_suffixes(per_source_suffixes[fn]), merged_name_cache
                )
                for fn in cc.foundation_names
            )
            manifest = embedding_manifest_fingerprint(
                SWEEP.embedding_root, target_foundations
            )
            train_aug_fp = train_aug_fingerprint_for(
                all_suffixes, ivy_source_names or cc.foundation_names
            )
        except Exception as exc:
            print(f"[warn] composite probe {cc.name!r} skipped: {exc}")
            continue

        foundation_label = label_with_train_augmentations(
            composite_foundation_label(target_foundations), all_suffixes
        )
        name = config_name(foundation_label, cc.name, cc.transform)
        cid = config_id(name, cc.estimator, target_foundations, manifest, train_aug_fp)
        if cid in terminal_ids and not SWEEP.overwrite:
            print(f"[skip] {cid} {name}: already registered")
            continue

        print(f"\nLoading composite embeddings: {foundation_label}")
        X_bundle, y, names, patients, slides, slide_groups = load_composite_embeddings(
            SWEEP.embedding_root,
            target_foundations,
            SWEEP.max_files,
            output_names=cc.foundation_names,  # keep estimator source keys stable/base-named
        )
        print(
            f"{foundation_label}: aligned patches={len(y):,}, "
            f"patients={len(np.unique(patients))}, slides={len(np.unique(slide_groups))}"
        )
        folds = build_folds(
            y,
            patients,
            slide_groups,
            GROUP_BY,
            N_SPLITS,
            SEED,
            force_train_patients=FORCE_TRAIN_PATIENTS,
        )
        patch_id, patient_id = resolve_patch_and_patient_ids(conn, patients, names)

        ivy_extras_by_source: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        ivy_summary: Dict[str, Any] | None = None
        if ivy_source_names:
            print(
                f"\nLoading train-only Ivy extras for composite sources: {list(ivy_source_names)}"
            )
            ivy_extras_by_source, ivy_summary = load_independent_ivy_source_extras(
                X_bundle,
                source_names=ivy_source_names,
            )
            for fn in sorted(ivy_extras_by_source):
                n_ivy = int(len(ivy_extras_by_source[fn][1]))
                sel = ivy_summary["sources"][fn]["selection"] if ivy_summary else {}
                print(
                    f"  {fn}: Ivy train-only extra rows = {n_ivy:,}; by class={sel.get('selected_by_class', {})}"
                )

        status = run_one_config(
            conn,
            oof_dir,
            cid,
            name,
            cc.transform,
            cc.estimator,
            X_bundle,
            y,
            folds,
            patch_id,
            patient_id,
            SWEEP.overwrite,
            source_train_extras=ivy_extras_by_source if ivy_source_names else None,
            train_aug_summary=ivy_summary if ivy_source_names else None,
        )
        if status == "completed":
            terminal_ids.add(cid)

        del X_bundle, y, names, patients, slides, slide_groups
        gc.collect()

    results = load_result_table(conn)
    ranked = print_ranking(results, n=50)
    print_per_class_f1(ranked, n=50)
    print(f"\nOOF npz files: {oof_dir}")
    print(f"Metrics registry: {out_dir / 'configs.db'}")
    conn.close()


if __name__ == "__main__":
    main()
