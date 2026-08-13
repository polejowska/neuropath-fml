
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


N_CLASSES = 10
CLASS_NAMES = {
    0: "CT",
    1: "DM",
    2: "IC",
    3: "LI",
    4: "MP",
    5: "NC",
    6: "PL",
    7: "PN",
    8: "WM",
    9: "NOTA",
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str = "") -> None:
    print(f"[{now()}] {message}", flush=True)


def ensure_finite(X: np.ndarray, label: str) -> np.ndarray:
    bad = ~np.isfinite(X).all(axis=1)
    if bad.any():
        log(f"[warn] {label}: zeroing {int(bad.sum()):,} rows containing NaN/Inf")
        X = X.copy()
        X[bad] = 0.0
    return X


def normalise_rows(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    sums = p.sum(axis=1, keepdims=True)
    bad = ~np.isfinite(sums[:, 0]) | (sums[:, 0] <= 0)
    if bad.any():
        p = p.copy()
        p[bad] = 1.0 / N_CLASSES
        sums = p.sum(axis=1, keepdims=True)
    p /= np.clip(sums, 1e-12, None)
    return p


def head_probability(model: Any, X: np.ndarray) -> np.ndarray:
    """Match the training/evaluation probability conversion for one saved head."""
    n = len(X)

    if hasattr(model, "predict_proba"):
        native = np.asarray(model.predict_proba(X), dtype=np.float32)
        native = np.nan_to_num(native, nan=0.0, posinf=1.0, neginf=0.0)
        classes = np.asarray(
            getattr(model, "classes_", np.arange(N_CLASSES)),
            dtype=np.int64,
        )
        out = np.zeros((n, N_CLASSES), dtype=np.float32)
        for j, class_id in enumerate(classes):
            if 0 <= int(class_id) < N_CLASSES and j < native.shape[1]:
                out[:, int(class_id)] = native[:, j]

    elif hasattr(model, "decision_function"):
        margins = np.asarray(model.decision_function(X))
        if margins.ndim == 1:
            margins = np.stack([-margins, margins], axis=1)

        classes = np.asarray(
            getattr(model, "classes_", np.arange(margins.shape[1])),
            dtype=np.int64,
        )
        scores = np.full((n, N_CLASSES), -20.0, dtype=np.float32)
        for j, class_id in enumerate(classes):
            if 0 <= int(class_id) < N_CLASSES and j < margins.shape[1]:
                scores[:, int(class_id)] = margins[:, j]

        z = scores.astype(np.float64)
        z -= z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        out = (
            exp_z / np.clip(exp_z.sum(axis=1, keepdims=True), 1e-12, None)
        ).astype(np.float32)

    else:
        pred = np.asarray(model.predict(X), dtype=np.int64)
        out = np.full((n, N_CLASSES), 1e-6, dtype=np.float32)
        out[np.arange(n), pred] = 1.0

    return normalise_rows(out)


def load_validation(
    embedding_root: Path,
    foundation: str,
    val_subpath: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = embedding_root / foundation / "val" / val_subpath
    if not path.is_file():
        raise FileNotFoundError(f"Validation embedding file not found: {path}")

    with np.load(path, allow_pickle=True) as d:
        required = {"X", "names"}
        missing = required - set(d.files)
        if missing:
            raise KeyError(f"{path}: missing arrays {sorted(missing)}")

        X = ensure_finite(
            np.asarray(d["X"], dtype=np.float32),
            f"{foundation}/val",
        )
        names = np.asarray(d["names"]).astype(str)
        y = (
            np.asarray(d["y"], dtype=np.int64)
            if "y" in d.files
            else np.full(len(names), -1, dtype=np.int64)
        )

    if not (len(X) == len(y) == len(names)):
        raise RuntimeError(
            f"{foundation}: X/y/names lengths differ: "
            f"{len(X)}, {len(y)}, {len(names)}"
        )
    return X, y, names


def apply_decision_rule(
    raw_probability: np.ndarray,
    rare_classes: np.ndarray,
    rare_boost: float,
    thresholds: np.ndarray,
) -> np.ndarray:
    p = np.asarray(raw_probability, dtype=np.float32).copy()

    if rare_boost != 1.0 and len(rare_classes):
        p[:, rare_classes] *= np.float32(rare_boost)
        p = normalise_rows(p)

    p /= np.clip(thresholds.reshape(1, -1), 1e-6, None)
    return normalise_rows(p)


def foundation_dimensions(
    manifest: Mapping[str, Any],
    foundations: tuple[str, ...],
) -> dict[str, int]:
    dimensions: dict[str, int] = {}
    for foundation in foundations:
        records = [
            record
            for record in manifest["heads"]
            if record["foundation"] == foundation
        ]
        if not records:
            raise RuntimeError(f"Manifest contains no heads for {foundation}")
        dimensions[foundation] = max(int(record["stop"]) for record in records)
    return dimensions


def exact_mask_for_foundation(
    *,
    foundation: str,
    sorted_foundations: tuple[str, ...],
    dimensions: Mapping[str, int],
    batch_rows: int,
    batch_start: int,
    batch_stop: int,
    pass_index: int,
    keep: float,
    seed: int,
) -> np.ndarray:
    """
    Recreate the exact internal-evaluation RNG stream for one foundation.

    The internal implementation creates one RNG per batch and, for each pass,
    generates masks for every foundation in sorted order. To remain
    foundation-sequential and memory efficient, this function deterministically
    replays that stream and returns only the requested mask.
    """
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(batch_start), int(batch_stop)])
    )

    target_mask: np.ndarray | None = None
    for current_pass in range(pass_index + 1):
        for source_name in sorted_foundations:
            shape = (batch_rows, int(dimensions[source_name]))
            random_values = rng.random(shape, dtype=np.float32)
            if current_pass == pass_index and source_name == foundation:
                target_mask = random_values < np.float32(keep)
                break
        if target_mask is not None:
            break

    if target_mask is None:
        raise RuntimeError(
            f"Failed to construct TTA mask for {foundation}, pass {pass_index}"
        )
    return target_mask


def predict_foundation(
    *,
    X: np.ndarray,
    foundation: str,
    foundation_heads: list[dict[str, Any]],
    model_dir: Path,
    batch_size: int,
    tta_aug: int,
    tta_keep: float,
    tta_seed: int,
    sorted_foundations: tuple[str, ...],
    dimensions: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    n_rows = len(X)
    clean_out = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    tta_out = (
        np.zeros((n_rows, N_CLASSES), dtype=np.float32)
        if tta_aug > 0
        else None
    )

    loaded_heads: list[tuple[dict[str, Any], Any]] = []
    for record in foundation_heads:
        head_path = model_dir / record["path"]
        if not head_path.is_file():
            raise FileNotFoundError(f"Missing saved head: {head_path}")
        log(f"loading head: {head_path.relative_to(model_dir)}")
        loaded_heads.append((record, joblib.load(head_path)))

    if not loaded_heads:
        raise RuntimeError(f"No heads loaded for {foundation}")

    for batch_start in tqdm(
        range(0, n_rows, batch_size),
        desc=f"predict {foundation}",
        unit="batch",
    ):
        batch_stop = min(batch_start + batch_size, n_rows)
        X_batch = X[batch_start:batch_stop]
        batch_rows = len(X_batch)

        clean_sum = np.zeros((batch_rows, N_CLASSES), dtype=np.float64)
        for record, model in loaded_heads:
            start = int(record["start"])
            stop = int(record["stop"])
            clean_sum += head_probability(model, X_batch[:, start:stop])

        clean_probability = normalise_rows(
            (clean_sum / float(len(loaded_heads))).astype(np.float32)
        )
        clean_out[batch_start:batch_stop] = clean_probability

        if tta_out is not None:
            augmented_sum = clean_probability.astype(np.float64)

            for pass_index in range(tta_aug):
                mask = exact_mask_for_foundation(
                    foundation=foundation,
                    sorted_foundations=sorted_foundations,
                    dimensions=dimensions,
                    batch_rows=batch_rows,
                    batch_start=batch_start,
                    batch_stop=batch_stop,
                    pass_index=pass_index,
                    keep=tta_keep,
                    seed=tta_seed,
                )
                masked_batch = (
                    X_batch * mask
                ) / np.float32(tta_keep)

                pass_sum = np.zeros((batch_rows, N_CLASSES), dtype=np.float64)
                for record, model in loaded_heads:
                    start = int(record["start"])
                    stop = int(record["stop"])
                    pass_sum += head_probability(
                        model,
                        masked_batch[:, start:stop],
                    )

                pass_probability = normalise_rows(
                    (pass_sum / float(len(loaded_heads))).astype(np.float32)
                )
                augmented_sum += pass_probability
                del mask, masked_batch, pass_sum, pass_probability

            tta_probability = normalise_rows(
                (augmented_sum / float(tta_aug + 1)).astype(np.float32)
            )
            tta_out[batch_start:batch_stop] = tta_probability

        del X_batch, clean_sum, clean_probability
        gc.collect()

    del loaded_heads
    gc.collect()
    return clean_out, tta_out


def save_outputs(
    *,
    prefix: Path,
    names: np.ndarray,
    y: np.ndarray,
    raw_probability: np.ndarray,
    final_probability: np.ndarray,
) -> tuple[Path, Path]:
    prediction = final_probability.argmax(axis=1).astype(np.int16)
    csv_path = prefix.with_suffix(".csv")
    npz_path = prefix.with_suffix(".npz")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "SubjectID": names.astype(str),
            "Prediction": prediction.astype(int),
        }
    ).to_csv(csv_path, index=False)

    np.savez_compressed(
        npz_path,
        names=names.astype(object),
        y=y.astype(np.int16),
        raw_proba=raw_probability.astype(np.float16),
        proba=final_probability.astype(np.float16),
        pred=prediction,
    )
    return csv_path, npz_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "official_models/"
            "vhg_aug_ivy14950_source_soft_full_exact_v1"
        ),
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path("artifacts/embeddings_by_patient_slide"),
    )
    parser.add_argument(
        "--val-subpath",
        default="patient-unmapped/slide-unmapped.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("official_predictions"),
    )
    parser.add_argument(
        "--output-stem",
        default="vhg_aug_ivy14950_source_soft",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--tta-aug",
        type=int,
        default=16,
        help="Number of masked TTA passes in addition to the clean pass.",
    )
    parser.add_argument("--tta-keep", type=float, default=0.90)
    parser.add_argument("--tta-seed", type=int, default=42)
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Write clean predictions only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output files to be replaced.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.tta_aug < 0:
        raise ValueError("--tta-aug must be non-negative")
    if not (0.0 < args.tta_keep <= 1.0):
        raise ValueError("--tta-keep must be in (0, 1]")

    tta_aug = 0 if args.no_tta else int(args.tta_aug)
    model_dir = args.model_dir.expanduser().resolve()
    embedding_root = args.embedding_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Training is not complete or manifest is missing: {manifest_path}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    foundations = tuple(str(x) for x in manifest["foundations"])
    sorted_foundations = tuple(sorted(foundations))
    configuration = manifest["configuration"]

    rare_classes = np.asarray(
        configuration["rare_classes"],
        dtype=np.int64,
    )
    rare_boost = float(configuration["rare_boost"])
    thresholds = np.asarray(
        configuration["thresholds"],
        dtype=np.float32,
    ).reshape(-1)

    if thresholds.shape != (N_CLASSES,):
        raise RuntimeError(
            f"Manifest contains invalid thresholds shape {thresholds.shape}"
        )
    if not np.isfinite(thresholds).all() or np.any(thresholds <= 0):
        raise RuntimeError("Manifest thresholds must be finite and positive")

    dimensions = foundation_dimensions(manifest, foundations)

    clean_prefix = out_dir / f"{args.output_stem}__clean"
    tta_prefix = out_dir / (
        f"{args.output_stem}__tta-dropout"
        f"{1.0 - args.tta_keep:g}x{tta_aug}"
    )
    expected_paths = [
        clean_prefix.with_suffix(".csv"),
        clean_prefix.with_suffix(".npz"),
        out_dir / f"{args.output_stem}__inference_summary.json",
    ]
    if tta_aug > 0:
        expected_paths.extend(
            [
                tta_prefix.with_suffix(".csv"),
                tta_prefix.with_suffix(".npz"),
            ]
        )
    existing = [path for path in expected_paths if path.exists()]
    if existing and not args.overwrite:
        raise RuntimeError(
            "Output files already exist. Use a new --out-dir/--output-stem "
            f"or pass --overwrite. Examples: {[str(p) for p in existing[:5]]}"
        )

    log("=" * 88)
    log("FINAL OFFICIAL INFERENCE")
    log(f"model directory: {model_dir}")
    log(f"embedding root: {embedding_root}")
    log(f"foundations: {foundations}")
    log(f"foundation dimensions: {dimensions}")
    log(f"saved heads: {manifest['n_heads']}")
    log(
        "rare classes: "
        f"{rare_classes.tolist()} = "
        f"{[CLASS_NAMES.get(int(c), str(int(c))) for c in rare_classes]}"
    )
    log(f"rare boost: {rare_boost}")
    log(f"thresholds: {thresholds.round(6).tolist()}")
    if tta_aug > 0:
        log(
            f"TTA enabled: 1 clean + {tta_aug} masked passes, "
            f"keep={args.tta_keep}, seed={args.tta_seed}, "
            f"batch={args.batch_size}"
        )
    else:
        log("TTA disabled: clean inference only")
    log("=" * 88)

    ensemble_clean: np.ndarray | None = None
    ensemble_tta: np.ndarray | None = None
    reference_names: np.ndarray | None = None
    reference_y: np.ndarray | None = None

    source_dir = out_dir / "source_probabilities"
    source_dir.mkdir(parents=True, exist_ok=True)

    for foundation_number, foundation in enumerate(foundations, start=1):
        log("")
        log("#" * 88)
        log(
            f"FOUNDATION {foundation_number}/{len(foundations)}: "
            f"{foundation}"
        )
        log("#" * 88)

        X, y, names = load_validation(
            embedding_root,
            foundation,
            args.val_subpath,
        )
        expected_dim = dimensions[foundation]
        if X.shape[1] != expected_dim:
            raise RuntimeError(
                f"{foundation}: validation dimension {X.shape[1]} "
                f"does not match trained head dimension {expected_dim}"
            )
        log(f"{foundation}: validation matrix shape={X.shape}")

        if reference_names is None:
            reference_names = names
            reference_y = y
            ensemble_clean = np.zeros(
                (len(names), N_CLASSES),
                dtype=np.float64,
            )
            if tta_aug > 0:
                ensemble_tta = np.zeros(
                    (len(names), N_CLASSES),
                    dtype=np.float64,
                )
        else:
            if not np.array_equal(names, reference_names):
                raise RuntimeError(
                    f"{foundation}: validation names/order differ from "
                    "the first foundation"
                )
            labelled = (y >= 0) & (reference_y >= 0)
            if labelled.any() and not np.array_equal(
                y[labelled],
                reference_y[labelled],
            ):
                raise RuntimeError(
                    f"{foundation}: validation labels differ from "
                    "the first foundation"
                )

        records = [
            record
            for record in manifest["heads"]
            if record["foundation"] == foundation
        ]
        records.sort(
            key=lambda record: (
                str(record.get("classifier", "")),
                int(record["start"]),
                int(record["stop"]),
                str(record["path"]),
            )
        )
        log(f"{foundation}: using {len(records)} saved heads")

        source_clean, source_tta = predict_foundation(
            X=X,
            foundation=foundation,
            foundation_heads=records,
            model_dir=model_dir,
            batch_size=args.batch_size,
            tta_aug=tta_aug,
            tta_keep=args.tta_keep,
            tta_seed=args.tta_seed,
            sorted_foundations=sorted_foundations,
            dimensions=dimensions,
        )

        ensemble_clean += source_clean
        np.save(
            source_dir / f"{foundation}__clean_raw.npy",
            source_clean.astype(np.float16),
        )

        if ensemble_tta is not None and source_tta is not None:
            ensemble_tta += source_tta
            np.save(
                source_dir / f"{foundation}__tta_raw.npy",
                source_tta.astype(np.float16),
            )

        del X, y, names, source_clean, source_tta
        gc.collect()
        log(f"{foundation}: probabilities saved; embedding matrix released")

    if (
        ensemble_clean is None
        or reference_names is None
        or reference_y is None
    ):
        raise RuntimeError("No predictions were produced")

    raw_clean = normalise_rows(
        (ensemble_clean / float(len(foundations))).astype(np.float32)
    )
    final_clean = apply_decision_rule(
        raw_clean,
        rare_classes,
        rare_boost,
        thresholds,
    )
    clean_csv, clean_npz = save_outputs(
        prefix=clean_prefix,
        names=reference_names,
        y=reference_y,
        raw_probability=raw_clean,
        final_probability=final_clean,
    )

    log("")
    log(f"clean CSV: {clean_csv}")
    log(f"clean NPZ: {clean_npz}")

    clean_counts = np.bincount(
        final_clean.argmax(axis=1),
        minlength=N_CLASSES,
    )
    log(
        "clean predicted counts: "
        + ", ".join(
            f"{CLASS_NAMES[c]}={int(clean_counts[c]):,}"
            for c in range(N_CLASSES)
        )
    )

    summary: dict[str, Any] = {
        "model_dir": str(model_dir),
        "manifest": str(manifest_path),
        "embedding_root": str(embedding_root),
        "val_subpath": args.val_subpath,
        "n_rows": int(len(reference_names)),
        "foundations": list(foundations),
        "n_heads": int(manifest["n_heads"]),
        "aggregation": "head mean within foundation, equal foundation mean",
        "rare_classes": rare_classes.astype(int).tolist(),
        "rare_boost": rare_boost,
        "thresholds": thresholds.astype(float).tolist(),
        "clean_csv": str(clean_csv),
        "clean_npz": str(clean_npz),
        "clean_prediction_counts": {
            CLASS_NAMES[c]: int(clean_counts[c])
            for c in range(N_CLASSES)
        },
        "tta_aug": int(tta_aug),
        "tta_keep": float(args.tta_keep),
        "tta_seed": int(args.tta_seed),
        "batch_size": int(args.batch_size),
    }

    if ensemble_tta is not None:
        raw_tta = normalise_rows(
            (ensemble_tta / float(len(foundations))).astype(np.float32)
        )
        final_tta = apply_decision_rule(
            raw_tta,
            rare_classes,
            rare_boost,
            thresholds,
        )
        tta_csv, tta_npz = save_outputs(
            prefix=tta_prefix,
            names=reference_names,
            y=reference_y,
            raw_probability=raw_tta,
            final_probability=final_tta,
        )

        clean_pred = final_clean.argmax(axis=1)
        tta_pred = final_tta.argmax(axis=1)
        probability_difference = np.abs(
            final_clean.astype(np.float64)
            - final_tta.astype(np.float64)
        )
        flips = int(np.sum(clean_pred != tta_pred))
        tta_counts = np.bincount(tta_pred, minlength=N_CLASSES)

        summary.update(
            {
                "tta_csv": str(tta_csv),
                "tta_npz": str(tta_npz),
                "tta_prediction_counts": {
                    CLASS_NAMES[c]: int(tta_counts[c])
                    for c in range(N_CLASSES)
                },
                "argmax_flips": flips,
                "argmax_agreement": float(
                    np.mean(clean_pred == tta_pred)
                ),
                "mean_absolute_probability_difference": float(
                    probability_difference.mean()
                ),
                "max_absolute_probability_difference": float(
                    probability_difference.max()
                ),
                "rows_with_any_probability_difference_gt_1e-6": int(
                    np.sum(
                        np.any(
                            probability_difference > 1e-6,
                            axis=1,
                        )
                    )
                ),
            }
        )

        log("")
        log(f"TTA CSV: {tta_csv}")
        log(f"TTA NPZ: {tta_npz}")
        log(
            "TTA predicted counts: "
            + ", ".join(
                f"{CLASS_NAMES[c]}={int(tta_counts[c]):,}"
                for c in range(N_CLASSES)
            )
        )
        log(
            f"clean/TTA argmax agreement: "
            f"{100.0 * summary['argmax_agreement']:.4f}% "
            f"({flips:,} changed predictions)"
        )
        log(
            "mean absolute probability difference: "
            f"{summary['mean_absolute_probability_difference']:.8f}"
        )
        log(
            "maximum absolute probability difference: "
            f"{summary['max_absolute_probability_difference']:.8f}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        out_dir / f"{args.output_stem}__inference_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    log("")
    log(f"inference summary: {summary_path}")
    log("INFERENCE COMPLETE")


if __name__ == "__main__":
    main()
