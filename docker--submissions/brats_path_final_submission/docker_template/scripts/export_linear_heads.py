#!/usr/bin/env python3
"""Export fitted sklearn linear heads to a version-neutral NumPy format.

Run this ONCE in the original training environment that can load the .joblib
files (for this submission: Python 3.13.13, scikit-learn 1.9.0, NumPy 2.5.1,
SciPy 1.18.0, joblib 1.5.3), before building the Docker image.

The script:
  * reads src/ckpts/manifest.json
  * loads each legacy .joblib head
  * verifies it is one of the supported fitted linear estimators
  * writes a compressed .npz containing only coef_, intercept_, classes_
  * self-checks the NumPy inference formula against sklearn on deterministic
    synthetic rows
  * updates manifest.json to point to the .npz files
  * preserves the original manifest as manifest.joblib.json

It does NOT delete the source .joblib files. .dockerignore excludes them from
the Docker build once the .npz files have been produced.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import sklearn
from scipy.special import expit


FORMAT_VERSION = "numpy_linear_v1"


def _stable_sigmoid(scores: np.ndarray) -> np.ndarray:
    return expit(np.asarray(scores))


def _numpy_decision(model: Any, X: np.ndarray) -> np.ndarray:
    return np.asarray(X) @ np.asarray(model.coef_).T + np.asarray(model.intercept_)


def _numpy_sgd_proba(model: Any, X: np.ndarray) -> np.ndarray:
    p = _stable_sigmoid(_numpy_decision(model, X))
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-300, None)


def _classify_supported_model(model: Any, path: Path) -> str:
    name = type(model).__name__

    if name == "SGDClassifier":
        if getattr(model, "loss", None) != "log_loss":
            raise RuntimeError(
                f"{path}: SGDClassifier has loss={getattr(model, 'loss', None)!r}; "
                "only loss='log_loss' is supported."
            )
        if not hasattr(model, "predict_proba"):
            raise RuntimeError(f"{path}: fitted SGDClassifier has no predict_proba().")
        return "sgd_log_loss"

    if name == "RidgeClassifier":
        return "ridge"

    raise RuntimeError(
        f"{path}: unsupported estimator type {type(model)!r}. "
        "This exporter intentionally supports only the fitted head types in this manifest."
    )


def _validate_shape(model: Any, expected_dim: int, path: Path) -> None:
    coef = np.asarray(model.coef_)
    intercept = np.asarray(model.intercept_)
    classes = np.asarray(model.classes_)

    if coef.ndim != 2 or intercept.ndim != 1 or classes.ndim != 1:
        raise RuntimeError(
            f"{path}: unexpected fitted parameter shapes: "
            f"coef={coef.shape}, intercept={intercept.shape}, classes={classes.shape}"
        )
    if coef.shape[1] != expected_dim:
        raise RuntimeError(
            f"{path}: coef feature dimension {coef.shape[1]} != manifest dimension {expected_dim}"
        )
    if coef.shape[0] != intercept.shape[0] or coef.shape[0] != classes.shape[0]:
        raise RuntimeError(
            f"{path}: class dimension mismatch: coef={coef.shape}, "
            f"intercept={intercept.shape}, classes={classes.shape}"
        )


def _self_check(
    model: Any, kind: str, expected_dim: int, path: Path, seed: int
) -> float:
    # Float32 mirrors the embedding chunk dtype passed by Docker inference.
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 0.1, size=(17, expected_dim)).astype(np.float32)

    if kind == "sgd_log_loss":
        expected = np.asarray(model.predict_proba(X), dtype=np.float64)
        actual = _numpy_sgd_proba(model, X)
    else:
        expected = np.asarray(model.decision_function(X), dtype=np.float64)
        actual = np.asarray(_numpy_decision(model, X), dtype=np.float64)

    max_abs = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=1e-11, atol=1e-12, equal_nan=True):
        raise RuntimeError(
            f"{path}: NumPy export self-check failed; max_abs_diff={max_abs:.3e}"
        )
    return max_abs


def export_heads(ckpts_dir: Path) -> None:
    ckpts_dir = ckpts_dir.resolve()
    manifest_path = ckpts_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest: Dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    backup_path = ckpts_dir / "manifest.joblib.json"
    if not backup_path.exists():
        shutil.copy2(manifest_path, backup_path)
        print(f"[export] Saved original manifest -> {backup_path}", flush=True)

    heads = manifest.get("heads", [])
    if int(manifest.get("n_heads", len(heads))) != len(heads):
        raise RuntimeError(
            f"manifest n_heads={manifest.get('n_heads')} but contains {len(heads)} head records"
        )

    print(
        f"[export] Environment: sklearn={sklearn.__version__}, "
        f"numpy={np.__version__}, joblib={joblib.__version__}",
        flush=True,
    )
    print(f"[export] Exporting {len(heads)} heads from {ckpts_dir}", flush=True)

    max_diff = 0.0
    for i, record in enumerate(heads, start=1):
        # On a rerun, source_joblib_path preserves the original source even
        # though record['path'] already points at .npz.
        source_rel = Path(record.get("source_joblib_path", record["path"]))
        if source_rel.suffix.lower() != ".joblib":
            raise RuntimeError(
                f"Head {i}: cannot locate legacy joblib source from {source_rel}. "
                "Restore manifest.joblib.json if necessary."
            )
        source_path = ckpts_dir / source_rel
        if not source_path.is_file():
            raise FileNotFoundError(f"Head {i}: missing source joblib: {source_path}")

        model = joblib.load(source_path)
        kind = _classify_supported_model(model, source_path)
        expected_dim = int(record["dimension"])
        _validate_shape(model, expected_dim, source_path)
        diff = _self_check(model, kind, expected_dim, source_path, seed=1000 + i)
        max_diff = max(max_diff, diff)

        out_rel = source_rel.with_suffix(".npz")
        out_path = ckpts_dir / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            format_version=np.asarray(FORMAT_VERSION),
            kind=np.asarray(kind),
            coef=np.asarray(model.coef_),
            intercept=np.asarray(model.intercept_),
            classes=np.asarray(model.classes_, dtype=np.int64),
        )

        # Validate the file without pickle/object arrays.
        with np.load(out_path, allow_pickle=False) as payload:
            if str(payload["format_version"].item()) != FORMAT_VERSION:
                raise RuntimeError(f"{out_path}: failed round-trip format validation")
            if tuple(payload["coef"].shape) != tuple(np.asarray(model.coef_).shape):
                raise RuntimeError(
                    f"{out_path}: failed round-trip coefficient validation"
                )

        record["source_joblib_path"] = source_rel.as_posix()
        record["path"] = out_rel.as_posix()
        record["format"] = FORMAT_VERSION
        record["exported_from_sklearn"] = sklearn.__version__

        print(
            f"[export] {i:02d}/{len(heads)} {kind:12s} "
            f"{source_rel.name} -> {out_rel.name} "
            f"(self-check max_abs={diff:.3e})",
            flush=True,
        )

    manifest["head_serialization"] = FORMAT_VERSION
    manifest["head_export_sklearn_version"] = sklearn.__version__
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"[export] Done. Updated {manifest_path}; "
        f"maximum self-check absolute difference={max_diff:.3e}",
        flush=True,
    )
    print(
        "[export] Legacy .joblib files were preserved locally but are excluded "
        "from Docker by .dockerignore.",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpts-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "ckpts",
        help="Checkpoint directory containing manifest.json and heads/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    export_heads(parse_args().ckpts_dir)
