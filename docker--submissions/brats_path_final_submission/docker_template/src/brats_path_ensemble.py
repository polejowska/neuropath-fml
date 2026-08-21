"""
brats_path_ensemble.py -- inference-only counterpart to
`BratsPath2025ChunkedSGDEnsemble` from the canonical training script
`genbiopathfm_ivy_end_to_end.py`.

WHY THIS FILE EXISTS: an earlier version of this Docker package's inference
code reimplemented the TTA/aggregation math independently (in
inference_dependencies.py), based on an assumption about what the real
training/eval script did. That reimplementation was NOT equivalent to the
real thing in two ways: (1) it masked and averaged each foundation's TTA
passes independently, then combined foundations only at the very end,
whereas the real script draws one shared mask across all foundations per
pass and combines them (source_mean) INSIDE each pass, before averaging
across passes; (2) it re-seeded a fresh RNG per mini-batch, whereas the real
script uses one continuous `np.random.default_rng(tta_seed)` advanced
sequentially across all TTA passes.

To eliminate any risk of a third subtly-wrong reimplementation, every method
below is copied VERBATIM (same expressions, same order of operations) from
`BratsPath2025ChunkedSGDEnsemble` in genbiopathfm_ivy_end_to_end.py --
specifically `_chunk_head_proba`, `_raw_avg_proba`, `_apply_rare_boost`,
`_apply_decision_rule`, `_mask_bundle`, `predict_proba`, `predict_proba_tta`,
`predict_tta`. The only things removed are what inference never needs:
`.fit()` and everything under it (chunk-head training, Ivy/stainaug loaders,
threshold optimization, sklearn BaseEstimator plumbing). Do not "clean up"
or restructure the methods kept below without diffing them against the
original class in genbiopathfm_ivy_end_to_end.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
from scipy.special import expit

N_CLASSES = 10
CLASS_IDS = np.arange(N_CLASSES, dtype=np.int64)


def _linear_decision_function(
    X: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
) -> np.ndarray:
    """Version-neutral equivalent of sklearn linear-estimator decision_function."""
    return np.asarray(X) @ coef.T + intercept


def _stable_sigmoid(scores: np.ndarray) -> np.ndarray:
    """Match sklearn's SGD log-loss probability path via scipy.special.expit."""
    return expit(np.asarray(scores))


class NumpySGDLogLossHead:
    """Inference-only representation of multiclass SGDClassifier(loss='log_loss').

    sklearn's multiclass SGD log-loss ``predict_proba`` applies a one-vs-rest
    sigmoid to each class margin, then normalizes the rows. Storing only
    coef_/intercept_/classes_ avoids sklearn/joblib pickle-version coupling.
    """

    def __init__(self, coef: np.ndarray, intercept: np.ndarray, classes: np.ndarray):
        self.coef_ = np.asarray(coef)
        self.intercept_ = np.asarray(intercept)
        self.classes_ = np.asarray(classes, dtype=np.int64)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return _linear_decision_function(X, self.coef_, self.intercept_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        if scores.ndim != 2:
            raise RuntimeError("Packaged SGD log-loss heads must be multiclass.")
        proba = _stable_sigmoid(scores)
        normalizer = proba.sum(axis=1, keepdims=True)
        # This mirrors sklearn's normalization while guarding pathological rows.
        return proba / np.clip(normalizer, 1e-300, None)


class NumpyRidgeHead:
    """Inference-only representation of a fitted multiclass RidgeClassifier."""

    def __init__(self, coef: np.ndarray, intercept: np.ndarray, classes: np.ndarray):
        self.coef_ = np.asarray(coef)
        self.intercept_ = np.asarray(intercept)
        self.classes_ = np.asarray(classes, dtype=np.int64)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return _linear_decision_function(X, self.coef_, self.intercept_)


def _load_numpy_linear_head(head_path: Path) -> Any:
    """Load a version-neutral NumPy linear head written by export_linear_heads.py."""
    if head_path.suffix.lower() != ".npz":
        raise RuntimeError(
            f"Legacy/non-NumPy head path in manifest: {head_path}. "
            "Run `python scripts/export_linear_heads.py` in the original "
            "Python 3.13 / scikit-learn 1.9 training environment before building Docker."
        )

    with np.load(head_path, allow_pickle=False) as payload:
        required = {"format_version", "kind", "coef", "intercept", "classes"}
        missing = required.difference(payload.files)
        if missing:
            raise RuntimeError(
                f"{head_path}: missing packaged head fields: {sorted(missing)}"
            )

        format_version = str(payload["format_version"].item())
        kind = str(payload["kind"].item())
        coef = np.asarray(payload["coef"])
        intercept = np.asarray(payload["intercept"])
        classes = np.asarray(payload["classes"], dtype=np.int64)

    if format_version != "numpy_linear_v1":
        raise RuntimeError(f"{head_path}: unsupported head format {format_version!r}")
    if coef.ndim != 2 or intercept.ndim != 1:
        raise RuntimeError(
            f"{head_path}: invalid coefficient shapes coef={coef.shape}, intercept={intercept.shape}"
        )
    if coef.shape[0] != intercept.shape[0] or coef.shape[0] != classes.shape[0]:
        raise RuntimeError(
            f"{head_path}: class dimension mismatch coef={coef.shape}, "
            f"intercept={intercept.shape}, classes={classes.shape}"
        )

    if kind == "sgd_log_loss":
        return NumpySGDLogLossHead(coef, intercept, classes)
    if kind == "ridge":
        return NumpyRidgeHead(coef, intercept, classes)
    raise RuntimeError(f"{head_path}: unsupported packaged linear-head kind {kind!r}")


# ── Verbatim from genbiopathfm_ivy_end_to_end.py ─────────────────────────────


def _chunk_head_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Return [N, N_CLASSES] float32 probabilities for a single chunk head.

    NaN-aware bad-row guard, matching the sweep's predict_proba_n: a NaN row-sum
    (which predict_proba can emit on numerical edge cases) is repaired to uniform,
    not just a non-positive one.

    Works for any estimator type: probability-native (predict_proba), margin-based
    (decision_function, e.g. RidgeClassifier / linear SVM), or predict-only.
    """
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
    # Catch NaN row-sums too, not just non-positive ones.
    bad = ~np.isfinite(row_sum[:, 0]) | (row_sum[:, 0] <= 0)
    out[bad] = 1.0 / N_CLASSES
    out[~bad] /= row_sum[~bad]
    return out


class FittedChunkEnsemble:
    """Inference-only counterpart to BratsPath2025ChunkedSGDEnsemble.

    Constructed directly from already-fitted heads (no .fit() call, no
    sklearn BaseEstimator machinery) -- heads are loaded from version-neutral
    NumPy .npz files via load_fitted_ensemble() below, using manifest.json for everything
    else (thresholds, rare_classes, rare_boost, source_aggregation).
    """

    def __init__(
        self,
        heads: List[Tuple[str, int, int, Any]],
        thresholds: np.ndarray,
        rare_classes: np.ndarray,
        rare_boost: float,
        source_weights: Mapping[str, float] | None = None,
        source_aggregation: str = "source_mean",
    ):
        self.heads_ = heads
        self.thresholds_ = np.asarray(thresholds, dtype=np.float32)
        self.rare_classes_ = np.asarray(rare_classes, dtype=np.int64)
        self.rare_boost = float(rare_boost)
        self.source_weights = source_weights
        self.source_aggregation = str(source_aggregation)

    # ── Verbatim from BratsPath2025ChunkedSGDEnsemble ────────────────────────

    def _source_weight(self, source_name: str) -> float:
        if self.source_weights is None:
            return 1.0
        try:
            return float(self.source_weights.get(source_name, 1.0))
        except AttributeError:
            return 1.0

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

    def _apply_decision_rule(self, raw_proba: np.ndarray) -> np.ndarray:
        """rare-boost → per-class threshold scaling → renormalize.

        This is exactly the tail of predict_proba(); factoring it out lets the
        clean and TTA paths apply an identical decision rule to their (possibly
        averaged) raw posteriors.
        """
        p = self._apply_rare_boost(raw_proba)
        p = p / np.clip(self.thresholds_.reshape(1, -1), 1e-6, None)
        p /= np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
        return p.astype(np.float32, copy=False)

    def predict_proba(self, X: Mapping[str, np.ndarray]) -> np.ndarray:
        return self._apply_decision_rule(self._raw_avg_proba(X))

    def predict(self, X: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1).astype(np.int64)

    @staticmethod
    def _mask_bundle(
        X: Mapping[str, np.ndarray],
        keep: float,
        rng: np.random.Generator,
    ) -> Dict[str, np.ndarray]:
        """One masked view of the embedding bundle.

        Per-dimension Bernoulli(keep) mask, kept dims rescaled by 1/keep
        (inverted dropout). Draws are taken on the FULL embedding before any
        chunk slicing happens downstream, so masking is chunk-respecting by
        construction. Sources are iterated in sorted order for reproducibility.
        """
        out: Dict[str, np.ndarray] = {}
        for src in sorted(X.keys()):
            a = np.asarray(X[src], dtype=np.float32)
            if keep >= 1.0:
                out[src] = a
                continue
            m = rng.random(a.shape, dtype=np.float32) < keep  # bool[N, dim]
            out[src] = (a * m) / np.float32(keep)  # float32, expectation-preserving
        return out

    def predict_proba_tta(
        self,
        X: Mapping[str, np.ndarray],
        tta_aug: int = 0,
        tta_keep: float = 0.9,
        tta_seed: int = 42,
    ) -> np.ndarray:
        """Posteriors with optional chunked feature-masking TTA.

        Averages RAW ensemble posteriors over (1 clean + tta_aug masked) passes,
        then applies the trained rare-boost + thresholds ONCE. With tta_aug <= 0
        this short-circuits to predict_proba(), i.e. bit-for-bit identical to the
        matched chunksgd_a3e-5_c768_source config (now augmented with any probe
        heads as configured).
        """
        if int(tta_aug) <= 0:
            return self.predict_proba(X)  # exact base config, no masking

        keep = float(tta_keep)
        if not (0.0 < keep <= 1.0):
            raise ValueError("tta_keep must be in (0, 1].")

        # Clean pass first, then masked passes; accumulate raw posteriors in f64.
        acc = self._raw_avg_proba(X).astype(np.float64)
        n_passes = 1
        rng = np.random.default_rng(int(tta_seed))
        for _ in range(int(tta_aug)):
            Xm = self._mask_bundle(X, keep, rng)
            acc += self._raw_avg_proba(Xm).astype(np.float64)
            n_passes += 1
            del Xm

        raw_mean = (acc / n_passes).astype(np.float32)
        return self._apply_decision_rule(raw_mean)

    def predict_tta(
        self,
        X: Mapping[str, np.ndarray],
        tta_aug: int = 0,
        tta_keep: float = 0.9,
        tta_seed: int = 42,
    ) -> np.ndarray:
        return (
            self.predict_proba_tta(X, tta_aug, tta_keep, tta_seed)
            .argmax(axis=1)
            .astype(np.int64)
        )


# ── Manifest / head loading (this Docker package's own code, not vendored) ──


def load_manifest(ckpts_dir: Path) -> Dict[str, Any]:
    manifest_path = ckpts_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found at {manifest_path}")
    print(f"[model] Loading manifest: {manifest_path}", flush=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        f"[model] Manifest output: foundations={manifest['foundations']}, "
        f"n_heads={manifest['n_heads']}",
        flush=True,
    )
    return manifest


def foundation_dimensions(manifest: Mapping[str, Any]) -> Dict[str, int]:
    if "dimensions" in manifest and manifest["dimensions"]:
        return {k: int(v) for k, v in manifest["dimensions"].items()}
    dims: Dict[str, int] = {}
    for foundation in manifest["foundations"]:
        records = [r for r in manifest["heads"] if r["foundation"] == foundation]
        if not records:
            raise RuntimeError(f"manifest.json has no heads for {foundation}")
        dims[foundation] = max(int(r["stop"]) for r in records)
    return dims


def load_fitted_ensemble(
    ckpts_dir: Path, manifest: Mapping[str, Any]
) -> FittedChunkEnsemble:
    """Load every head listed in manifest.json and assemble a
    FittedChunkEnsemble ready to score a {foundation: embeddings} bundle,
    using the EXACT same math as the real, validated training/eval script."""
    cfg = manifest["configuration"]
    print(
        f"[model] Loading {manifest['n_heads']} head(s) across "
        f"{len(manifest['foundations'])} foundation(s) "
        f"(source_aggregation={cfg['source_aggregation']!r}).",
        flush=True,
    )
    heads: List[Tuple[str, int, int, Any]] = []
    for i, record in enumerate(manifest["heads"], start=1):
        head_path = ckpts_dir / record["path"]
        if not head_path.is_file():
            raise FileNotFoundError(f"Missing packaged head: {head_path}")
        clf = _load_numpy_linear_head(head_path)
        heads.append(
            (record["foundation"], int(record["start"]), int(record["stop"]), clf)
        )
        if i == 1 or i == manifest["n_heads"]:
            print(
                f"[model] Loaded head {i}/{manifest['n_heads']}: {record['path']}",
                flush=True,
            )

    ensemble = FittedChunkEnsemble(
        heads=heads,
        thresholds=np.asarray(cfg["thresholds"], dtype=np.float32),
        rare_classes=np.asarray(cfg["rare_classes"], dtype=np.int64),
        rare_boost=float(cfg["rare_boost"]),
        source_weights=None,
        source_aggregation=str(cfg["source_aggregation"]),
    )
    print(
        f"[model] Ensemble ready: rare_classes={ensemble.rare_classes_.tolist()}, "
        f"rare_boost={ensemble.rare_boost}, thresholds={ensemble.thresholds_.round(4).tolist()}",
        flush=True,
    )
    return ensemble
