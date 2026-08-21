from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ─────────────────────────────────────────────────────────────────

SEED = 42
N_CLASSES = 10
CLASS_IDS = np.arange(N_CLASSES, dtype=np.int64)
FOUNDATION_NAMES = ("virchow2", "hoptimus1", "genbiopathfm")

SCRIPT_VERSION = (
    "v_h_g_independent_ivy_plus_stainaug_and_stainaug_local_train_augmentation_v3"
    "_plus_allfoundations_ridge_lsqr_a10_probe"
)

# ── SGD base estimator (identical to sweep's _sgd) ────────────────────────────


def _sgd(
    alpha: float = 3e-5, loss: str = "log_loss", max_iter: int = 20
) -> SGDClassifier:
    return SGDClassifier(
        loss=loss,
        alpha=alpha,
        penalty="l2",
        max_iter=max_iter,
        tol=1e-3,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=1,
        early_stopping=True,
        validation_fraction=0.05,
        n_iter_no_change=3,
    )


# ── Additional probe estimator: Ridge (lsqr, l2, balanced) ────────────────────
#
# Originally mirrored the sweep's virchow2-only probe:
#   ProbeConfig(
#       "ridge_lsqr_a10_l2_balanced",
#       RidgeClassifier(alpha=10.0, class_weight="balanced", solver="lsqr",
#                        tol=1e-2, max_iter=50),
#       "l2",
#       only_foundations=("virchow2",),
#       aug_artifact_suffixes=("ivy", "stainaug", "stainaug_local"),
#   )
#
# This has now been generalized: the same Ridge probe config can be attached
# to ANY subset of foundations (by default, all of them), each getting its own
# independently-fit set of Ridge chunk heads. Every probe is fit on exactly
# the same per-chunk, per-source training rows used for the primary SGD heads
# for that source (base BraTS rows + whatever independent Ivy GAP / stainaug /
# stainaug_local extras were assembled for that source upstream) -- no
# separate data loading path is required, since the extras are already merged
# per-foundation before heads are fit.


def _ridge_lsqr_a10_l2_balanced() -> RidgeClassifier:
    return RidgeClassifier(
        alpha=10.0,
        class_weight="balanced",
        solver="lsqr",
        tol=1e-2,
        max_iter=50,
    )


# ── Per-head probability helper (matches sweep's predict_proba_n / _chunk_head_proba) ──


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


# ── BratsPath2025ChunkedSGDEnsemble (matches sweep implementation) ────────────


class BratsPath2025ChunkedSGDEnsemble(BaseEstimator, ClassifierMixin):
    """
    Chunked SGD ensemble across multiple foundations, with optional additional
    "probe" estimators layered on top of the primary chunk-SGD heads for one
    or more sources (e.g. an extra RidgeClassifier probe for any subset of
    foundations, up to and including all of them).

    Accepts a dict X = {foundation_name: np.ndarray[N, dim]} for fit/predict.
    Each foundation's embedding is split into fixed-size chunks; one balanced
    SGDClassifier is trained per chunk (the primary estimator). Optionally,
    additional named probe estimators (any sklearn-compatible classifier) are
    trained on the SAME chunks / same fit rows for a configurable subset of
    sources, and their heads are appended into that source's head pool.
    Because source_mean aggregation averages all heads sharing a source_name
    together before averaging across sources, probe heads simply enrich that
    source's internal ensemble -- no changes to aggregation, calibration, or
    inference plumbing are required to add a probe to any (or every) source.

    Probabilities are averaged across all heads (head_mean) or first averaged
    within each source then across sources (source_mean). Optional rare-class
    boosting and per-class threshold scaling.

    Test-time augmentation (chunked feature masking) is available via
    predict_proba_tta(); it does not affect fit() and reduces exactly to
    predict_proba() when tta_aug <= 0.
    """

    def __init__(
        self,
        base_estimator: Any | None = None,
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
        probe_estimators: Mapping[str, Sequence[Tuple[str, Any]]] | None = None,
        verbose: int = 1,
        random_state: int = SEED,
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
        # Mapping: source_name -> [(probe_name, estimator_prototype), ...]
        self.probe_estimators = probe_estimators
        self.verbose = verbose
        self.random_state = random_state

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _iter_chunks(self, dim: int):
        step = int(self.chunk_size)
        if step <= 0:
            raise ValueError("chunk_size must be positive")
        made = False
        for start in range(0, int(dim), step):
            stop = min(start + step, int(dim))
            if stop - start >= int(self.min_chunk_dim):
                made = True
                yield int(start), int(stop)
        if not made and int(dim) > 0:
            yield 0, int(dim)

    def _new_estimator(
        self, seed_offset: int = 0, base_estimator: Any | None = None
    ) -> Any:
        """Clone an estimator prototype (defaults to self.base_estimator / SGD)
        and, if it exposes a random_state parameter, offset it deterministically
        so distinct chunk/probe heads don't share identical randomness."""
        proto = base_estimator if base_estimator is not None else self.base_estimator
        est = clone(proto) if proto is not None else _sgd(3e-5, "log_loss")
        params = getattr(est, "get_params", lambda: {})()
        if "random_state" in params:
            est.set_params(random_state=int(self.random_state) + int(seed_offset))
        return est

    def _fit_chunk_estimator(
        self,
        estimator_prototype: Any | None,
        Xchunk: np.ndarray,
        y_source: np.ndarray,
        sw: np.ndarray | None,
        seed_offset: int,
    ) -> Any:
        """Fit one chunk head, with the retry-without-early-stopping safety net.

        Shared by both the primary (SGD) chunk heads and any additional probe
        heads (e.g. Ridge) so both paths get identical failure handling.
        """
        clf = self._new_estimator(
            seed_offset=seed_offset, base_estimator=estimator_prototype
        )
        try:
            if sw is None:
                clf.fit(Xchunk, y_source)
            else:
                try:
                    clf.fit(Xchunk, y_source, sample_weight=sw)
                except TypeError:
                    clf.fit(Xchunk, y_source)
        except ValueError as exc:
            # Retry without early_stopping if the source-specific internal
            # validation split is too small after augmentation/subsampling.
            # (No-op for estimators, like RidgeClassifier, that lack this param.)
            params = getattr(clf, "get_params", lambda: {})()
            if params.get("early_stopping", False) and hasattr(clf, "set_params"):
                clf = self._new_estimator(
                    seed_offset=seed_offset, base_estimator=estimator_prototype
                )
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
        return clf

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
            n_splits=1,
            test_size=frac,
            random_state=int(self.random_state),
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

        In the base configuration every source receives the same BraTS fitting
        partition. With independent per-source extras (Ivy GAP, partial stain
        augmentation, or both), each source receives its own extra labelled
        rows -- possibly a different amount for each foundation. This is valid
        for source_mean aggregation because heads are trained and averaged per
        source; no source-level training-row alignment is required after the
        shared BraTS calibration split is defined.

        After the primary (self.base_estimator) heads are fit for every source,
        any configured `probe_estimators` are fit on the exact same per-source
        fit rows (same chunks, same X/y, same extras already merged in) and
        their heads are appended under the SAME source_name. Because
        source_mean averages all heads sharing a source_name before averaging
        across sources, a probe simply enriches that source's internal
        ensemble -- it needs no special-casing anywhere else in the class.
        `probe_estimators` may cover any subset of sources, including all of
        them at once.
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
                desc=f"fit chunk-SGD heads  n={len(jobs)}  source_rows={source_rows}",
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
            clf = self._fit_chunk_estimator(
                None,
                Xchunk,
                y_source,
                sw,
                seed_offset=seed_offset,
            )
            heads.append((source_name, int(start), int(stop), clf))

        # ── Additional probe estimators (e.g. Ridge) for selected sources ──────
        self.probe_heads_ = []
        probes_by_source = dict(self.probe_estimators or {})
        if probes_by_source:
            probe_jobs = []
            for source_name, probes in probes_by_source.items():
                if source_name not in X:
                    continue
                dim = int(np.asarray(X[source_name]).shape[1])
                for probe_name, estimator_prototype in probes:
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

            probe_source_rows = {
                name: source_rows[name] for name in sorted({j[1] for j in probe_jobs})
            }
            probe_iterator = (
                tqdm(
                    probe_jobs,
                    desc=f"fit probe heads  n={len(probe_jobs)}  source_rows={probe_source_rows}",
                    leave=False,
                )
                if int(self.verbose) > 0
                else probe_jobs
            )

            # Large, disjoint seed space so probe heads never collide with the
            # primary heads' random_state offsets.
            for i, (
                probe_name,
                source_name,
                start,
                stop,
                estimator_prototype,
            ) in enumerate(probe_iterator):
                Xchunk = np.asarray(X[source_name], dtype=np.float32)[:, start:stop]
                y_source = np.asarray(y_by_source[source_name], dtype=np.int64)
                sw = (
                    compute_sample_weight("balanced", y_source)
                    if bool(self.use_sample_weight)
                    else None
                )
                clf = self._fit_chunk_estimator(
                    estimator_prototype,
                    Xchunk,
                    y_source,
                    sw,
                    seed_offset=1_000_000 + i,
                )
                heads.append((source_name, start, stop, clf))
                self.probe_heads_.append(
                    {
                        "probe_name": probe_name,
                        "source": source_name,
                        "start": start,
                        "stop": stop,
                    }
                )

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

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X: Mapping[str, np.ndarray],
        y: np.ndarray,
        source_train_extras: Mapping[str, Tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        """Fit on aligned BraTS rows, optionally with independent per-source extras.

        The base BraTS split drives rare-class detection and threshold calibration
        exactly as before. ``source_train_extras`` is used only for chunk-head
        fitting; it is never visible to the BraTS-only calibration partition.
        Extras for a given source may come from any combination of sources
        (e.g. Ivy GAP rows concatenated with partial stain-augmentation rows) and
        different foundations may carry a different number of extra rows -- there
        is no requirement that every foundation's extras be the same size.

        Any configured ``self.probe_estimators`` are fit on the same per-source
        fit rows (including that source's extras) right after the primary heads,
        and their heads are appended into the same source bucket. This applies
        independently to every source named in ``self.probe_estimators``.
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
        # Preserve the base decision policy: rare classes are inferred from
        # original aligned BraTS labels, not source-specific extras.
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
                        f"{source_name}: extra X/y lengths disagree: "
                        f"{len(X_extra)} vs {len(y_extra)}."
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

        # Calibration stays strictly on the original BraTS calibration partition,
        # preserving the prior threshold-selection protocol. Because probe heads
        # are already part of self.heads_ by this point, p_cal (via
        # _raw_avg_proba) automatically reflects the blended SGD+probe ensemble.
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

    # ── Decision-rule tail shared by clean + TTA paths ────────────────────────

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

    # ── Test-time augmentation: chunked feature masking ───────────────────────

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
        tta_seed: int = SEED,
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
        tta_seed: int = SEED,
    ) -> np.ndarray:
        return (
            self.predict_proba_tta(X, tta_aug, tta_keep, tta_seed)
            .argmax(axis=1)
            .astype(np.int64)
        )


# ── Data utilities ────────────────────────────────────────────────────────────


def scrub(X: np.ndarray) -> np.ndarray:
    """Zero out rows that contain NaN or Inf (matches sweep load behaviour)."""
    bad = ~np.isfinite(X).all(axis=1)
    if bad.any():
        n = int(bad.sum())
        pct = 100.0 * n / max(len(X), 1)
        print(f"  [warn] {n:,} patches ({pct:.3f}%) contain NaN/Inf — zeroing out.")
        X = X.copy()
        X[bad] = 0.0
    return X


def load_foundation_train(
    root: Path, foundation: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load consolidated train embeddings for one foundation. Only y >= 0 kept."""
    base = root / foundation / "train"
    files = sorted(base.glob("patient-*/slide-*.npz"))
    if not files:
        raise RuntimeError(f"No consolidated train .npz files found under {base}")

    Xs, ys, names_list, patients_list = [], [], [], []
    for fp in tqdm(files, desc=f"load train/{foundation}", leave=True):
        d = np.load(fp, allow_pickle=True)
        y = d["y"].astype(np.int64)
        mask = y >= 0
        if not mask.any():
            continue
        Xs.append(d["X"].astype(np.float32)[mask])
        ys.append(y[mask])
        names_list.append(d["names"].astype(str)[mask])
        patients_list.extend([fp.parent.name] * int(mask.sum()))

    if not Xs:
        raise RuntimeError(
            f"All train patches for {foundation} have y=-1; cannot train."
        )

    X = scrub(np.concatenate(Xs, axis=0))
    y = np.concatenate(ys, axis=0)
    names = np.concatenate(names_list, axis=0).astype(str)
    patients = np.asarray(patients_list, dtype=object)
    return X, y, names, patients


def load_composite_train(
    root: Path,
    foundation_names: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Load + align train embeddings for multiple foundations by (patient, name)."""
    loaded: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for fn in foundation_names:
        loaded[fn] = load_foundation_train(root, fn)
        X, y, names, patients = loaded[fn]
        print(
            f"  {fn}: X={X.shape}  labels={np.bincount(y, minlength=N_CLASSES).tolist()}"
        )

    first = foundation_names[0]
    ref_X, ref_y, ref_names, ref_patients = loaded[first]

    # ── Fast path: identical patch order across all foundations ───────────────
    def same_meta(a, b):
        _, ya, na, pa = a
        _, yb, nb, pb = b
        return (
            len(ya) == len(yb)
            and np.array_equal(ya, yb)
            and np.array_equal(na, nb)
            and np.array_equal(pa, pb)
        )

    if all(same_meta(loaded[first], loaded[fn]) for fn in foundation_names[1:]):
        print("  [align] all foundations share the same patch order — fast path.")
        X_bundle = {fn: loaded[fn][0] for fn in foundation_names}
        return X_bundle, ref_y, ref_names, ref_patients

    # ── Slow path: align by (patient, patch-name) intersection ───────────────
    print("  [align] patch order differs; aligning by (patient, name) intersection.")
    ref_keys = [(str(p), str(n)) for p, n in zip(ref_patients, ref_names)]
    member_maps: Dict[str, Dict[Tuple[str, str], int]] = {}
    for fn in foundation_names[1:]:
        _, _, fn_names, fn_patients = loaded[fn]
        member_maps[fn] = {
            (str(p), str(n)): i for i, (p, n) in enumerate(zip(fn_patients, fn_names))
        }

    keep_ref_idx: list = []
    aligned_other_idx: Dict[str, list] = {fn: [] for fn in foundation_names[1:]}
    for i, k in enumerate(ref_keys):
        if all(k in member_maps[fn] for fn in foundation_names[1:]):
            keep_ref_idx.append(i)
            for fn in foundation_names[1:]:
                aligned_other_idx[fn].append(member_maps[fn][k])

    if not keep_ref_idx:
        raise RuntimeError(
            f"Composite {foundation_names}: no overlapping (patient, name) keys found."
        )

    keep_ref_idx_arr = np.asarray(keep_ref_idx, dtype=np.int64)
    X_bundle: Dict[str, np.ndarray] = {first: ref_X[keep_ref_idx_arr]}
    for fn in foundation_names[1:]:
        idx = np.asarray(aligned_other_idx[fn], dtype=np.int64)
        X_bundle[fn] = loaded[fn][0][idx]
        y_check = loaded[fn][1][idx]
        if not np.array_equal(ref_y[keep_ref_idx_arr], y_check):
            raise RuntimeError(
                f"Composite: labels disagree after alignment for {fn} vs {first}."
            )

    n_kept = len(keep_ref_idx_arr)
    n_ref = len(ref_y)
    print(f"  [align] kept {n_kept:,}/{n_ref:,} patches from reference ({first}).")
    return (
        X_bundle,
        ref_y[keep_ref_idx_arr],
        ref_names[keep_ref_idx_arr],
        ref_patients[keep_ref_idx_arr],
    )


# ── Ivy GAP independent source-specific training augmentation ───────────────

IVY_SUPPORTED_CLASS_IDS = (0, 2, 4, 5, 7)  # CT, IC, MP, NC, PN


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


def _read_ivy_embedding_config(
    ivy_root: Path, expected_foundation: str
) -> Dict[str, Any]:
    path = ivy_root / "embedding_config.json"
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
                f"{fp}: X/label_int/patch_uid lengths disagree: "
                f"{len(X)}, {len(y)}, {len(uid)}."
            )
        if not np.isfinite(X).all():
            raise ValueError(f"{fp}: non-finite Ivy embeddings detected.")
        unsupported = sorted(set(np.unique(y).tolist()) - set(IVY_SUPPORTED_CLASS_IDS))
        if unsupported:
            raise ValueError(
                f"{fp}: unexpected Ivy labels {unsupported}; expected only "
                f"{list(IVY_SUPPORTED_CLASS_IDS)}."
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
            f"{ivy_root}: found {int(repeated.sum()):,} duplicate patch_uid value(s), "
            f"e.g. {preview}. Resolve the duplicate block state before training."
        )

    return X_all, y_all, uid_all, cfg


def _select_independent_ivy_rows(
    X: np.ndarray,
    y: np.ndarray,
    source_name: str,
    max_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Sample one source's Ivy pool without looking at any other source.

    Distinct source-specific seeds intentionally avoid forcing Virchow2,
    H-optimus-1, and GenBioPathFM to select identical subsets even when their
    candidate pools happen to contain overlapping patch IDs.
    """
    max_per_class = int(max_per_class)
    if max_per_class < 0:
        raise ValueError(
            "--ivy-max-per-class must be >= 0; 0 means all rows per class."
        )

    source_offset = {
        "virchow2": 0,
        "hoptimus1": 10_000,
        "genbiopathfm": 20_000,
    }.get(source_name, 30_000)
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
                    f"but --ivy-max-per-class={max_per_class:,} was requested. "
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
        raise ValueError("--ivy-norm-mode must be one of: auto, l2, none.")
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
    virchow2_root: Path,
    hoptimus1_root: Path,
    genbiopathfm_root: Path,
    max_per_class: int,
    seed: int,
    norm_mode: str,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    """Return independent Ivy extras for all foundations, with no UID intersection."""
    required_sources = set(FOUNDATION_NAMES)
    if not required_sources.issubset(X_brats):
        raise RuntimeError(
            f"BraTS source bundle lacks required Ivy sources: {sorted(required_sources - set(X_brats))}."
        )

    roots = {
        "virchow2": Path(virchow2_root),
        "hoptimus1": Path(hoptimus1_root),
        "genbiopathfm": Path(genbiopathfm_root),
    }
    extras: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    source_summary: Dict[str, Any] = {}

    for source_name in FOUNDATION_NAMES:
        X_all, y_all, uid_all, cfg = _load_ivy_foundation_blocks(
            roots[source_name], source_name
        )
        X_selected, y_selected, selection = _select_independent_ivy_rows(
            X_all,
            y_all,
            source_name=source_name,
            max_per_class=max_per_class,
            seed=seed,
        )
        X_matched, representation = _match_one_ivy_source_to_brats(
            X_brats[source_name],
            X_selected,
            source_name=source_name,
            mode=norm_mode,
        )
        extras[source_name] = (X_matched, y_selected)
        source_summary[source_name] = {
            "root": str(roots[source_name].expanduser().resolve()),
            "model_key": str(cfg.get("model_key", "")),
            "embedding_dim": int(cfg.get("embedding_dim", X_all.shape[1])),
            "n_unique_rows_before_selection": int(len(uid_all)),
            "selection": selection,
            "representation_matching": representation,
        }

    return extras, {
        "augmentation_mode": "independent_per_source_no_uid_intersection",
        "ivy_max_per_class": int(max_per_class),
        "ivy_seed": int(seed),
        "sources": source_summary,
    }


# ── Stain-augmentation source-specific training extras ──────────────────────
#
# This mirrors the sweep's multi-suffix augmentation semantics:
#
#   aug_artifact_suffixes=("stainaug", "stainaug_local")
#
# Each suffix is resolved independently for each foundation. Search order also
# matches the sweep:
#   1. completed in-progress shards under
#      <artifacts_root>/embedding_parts/<foundation>_<suffix>/<split>/
#   2. consolidated files under
#      <embedding_root>/<foundation>_<suffix>/<split>/
#
# Rows from every available suffix are concatenated into that foundation's
# train-only extra pool. Across suffixes, both views are intentionally retained,
# even when they originate from the same clean patch. Calibration and inference
# remain on the original aligned BraTS rows only.


def _source_artifact(foundation: str, suffix: str) -> str:
    """Map 'stainaug_local' to '<foundation>_stainaug_local'."""
    suffix = str(suffix).strip()
    if not suffix:
        raise ValueError("Empty stain-augmentation suffix is not allowed.")
    return suffix if suffix.startswith(f"{foundation}_") else f"{foundation}_{suffix}"


def _find_completed_stainaug_shard_groups(
    parts_root: Path,
) -> Dict[Tuple[str, str], List[Path]]:
    groups: Dict[Tuple[str, str], List[Path]] = {}
    for shard_dir in sorted(parts_root.glob("shard-*")):
        if not (shard_dir / "_SHARD_DONE.json").exists():
            continue
        for part in shard_dir.glob("patient-*/slide-*.part.npz"):
            key = (part.parent.name, part.stem.replace(".part", ""))
            groups.setdefault(key, []).append(part)
    return groups


def _find_consolidated_stainaug_groups(
    final_root: Path,
) -> Dict[Tuple[str, str], List[Path]]:
    groups: Dict[Tuple[str, str], List[Path]] = {}
    for fp in sorted(final_root.glob("patient-*/slide-*.npz")):
        key = (fp.parent.name, fp.stem)
        groups.setdefault(key, []).append(fp)
    return groups


def _load_stainaug_file_group(
    paths: List[Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one (patient, slide) group and deduplicate by patch name.

    The stable sort + keep-last rule is the same rule used by the sweep and by
    the augmentation consolidator. Deduplication is only within one suffix;
    rows from stainaug and stainaug_local are both retained later.
    """
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    names: List[Any] = []
    for path in sorted(paths):
        with np.load(path, allow_pickle=True) as d:
            Xs.append(np.asarray(d["X"]))
            ys.append(np.asarray(d["y"]))
            names.extend(np.asarray(d["names"], dtype=object).tolist())

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    names_arr = np.asarray(names, dtype=object)

    order = np.argsort(names_arr.astype(str), kind="mergesort")
    X, y, names_arr = X[order], y[order], names_arr[order]
    if len(names_arr) > 1:
        _, unique_last_rev = np.unique(names_arr[::-1].astype(str), return_index=True)
        keep = np.sort(len(names_arr) - 1 - unique_last_rev)
        X, y, names_arr = X[keep], y[keep], names_arr[keep]
    return X, y, names_arr


def _resolve_stainaug_groups(
    artifacts_root: Path,
    embedding_root: Path,
    foundation: str,
    aug_suffix: str,
    split: str,
) -> Tuple[str, Path, Dict[Tuple[str, str], List[Path]]]:
    """Resolve one foundation/suffix using the sweep's search order."""
    artifact = _source_artifact(foundation, aug_suffix)
    parts_root = artifacts_root / "embedding_parts" / artifact / split
    final_root = embedding_root / artifact / split

    if parts_root.exists():
        groups = _find_completed_stainaug_shard_groups(parts_root)
        if groups:
            return "completed_parts", parts_root, groups

    if final_root.exists():
        groups = _find_consolidated_stainaug_groups(final_root)
        if groups:
            return "consolidated", final_root, groups

    return "missing", parts_root, {}


def _load_one_foundation_stainaug_extra(
    artifacts_root: Path,
    embedding_root: Path,
    foundation: str,
    aug_suffix: str,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load every available row for one foundation/suffix pair."""
    source_kind, source_root, groups = _resolve_stainaug_groups(
        artifacts_root=artifacts_root,
        embedding_root=embedding_root,
        foundation=foundation,
        aug_suffix=aug_suffix,
        split=split,
    )
    artifact = _source_artifact(foundation, aug_suffix)
    if not groups:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            {
                "artifact": artifact,
                "suffix": str(aug_suffix),
                "source_kind": source_kind,
                "source_root": str(source_root),
                "found": False,
                "n_slide_files": 0,
                "n_rows_before_label_filter": 0,
                "n_rows": 0,
                "n_dropped_negative_labels": 0,
                "per_class_counts": {},
            },
        )

    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for key in sorted(groups):
        X, y, _names = _load_stainaug_file_group(groups[key])
        Xs.append(X)
        ys.append(y)

    X_all = np.concatenate(Xs, axis=0).astype(np.float32, copy=False)
    y_all = np.concatenate(ys, axis=0).astype(np.int64, copy=False)
    n_before = int(len(y_all))

    valid = y_all >= 0
    n_dropped = int((~valid).sum())
    if n_dropped:
        X_all = X_all[valid]
        y_all = y_all[valid]
    X_all = scrub(X_all)

    per_class = (
        {int(c): int(n) for c, n in zip(*np.unique(y_all, return_counts=True))}
        if len(y_all)
        else {}
    )
    summary = {
        "artifact": artifact,
        "suffix": str(aug_suffix),
        "source_kind": source_kind,
        "source_root": str(source_root),
        "found": True,
        "n_slide_files": int(len(groups)),
        "n_rows_before_label_filter": n_before,
        "n_rows": int(len(y_all)),
        "n_dropped_negative_labels": n_dropped,
        "per_class_counts": per_class,
    }
    return X_all, y_all, summary


def _normalize_stainaug_suffixes(values: Sequence[str]) -> Tuple[str, ...]:
    """Normalize nargs/legacy CLI values, accepting comma-separated entries."""
    out: List[str] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token and token not in out:
                out.append(token)
    if not out:
        raise ValueError("At least one stain-augmentation suffix is required.")
    return tuple(out)


def load_independent_stainaug_extras(
    X_brats: Mapping[str, np.ndarray],
    foundation_names: Sequence[str],
    artifacts_root: Path,
    embedding_root: Path,
    aug_suffixes: Sequence[str],
    split: str,
    max_rows_per_foundation: int,
    seed: int,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    """Load and concatenate stainaug + stainaug_local per foundation.

    Foundations and suffixes are independent: a missing pool contributes zero
    rows and never blocks another foundation or suffix. The optional cap is
    applied after all configured suffix pools have been concatenated for a
    foundation, preserving the CLI's "per foundation" meaning.
    """
    suffixes = _normalize_stainaug_suffixes(tuple(aug_suffixes))
    artifacts_root = Path(artifacts_root).expanduser().resolve()
    embedding_root = Path(embedding_root).expanduser().resolve()

    per_foundation_parts: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
        foundation: [] for foundation in foundation_names
    }
    pool_summaries: Dict[str, Any] = {}

    for suffix in suffixes:
        suffix_sources: Dict[str, Any] = {}
        for foundation in foundation_names:
            X_extra, y_extra, summary = _load_one_foundation_stainaug_extra(
                artifacts_root=artifacts_root,
                embedding_root=embedding_root,
                foundation=foundation,
                aug_suffix=suffix,
                split=split,
            )
            if len(y_extra) > 0:
                expected_dim = int(np.asarray(X_brats[foundation]).shape[1])
                if X_extra.ndim != 2 or X_extra.shape[1] != expected_dim:
                    raise ValueError(
                        f"{foundation}/{suffix}: augmentation shape {X_extra.shape} is "
                        f"incompatible with BraTS feature dim {expected_dim}."
                    )
                per_foundation_parts[foundation].append((X_extra, y_extra))
            suffix_sources[foundation] = summary
        pool_summaries[suffix] = {"sources": suffix_sources}

    extras: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    source_summaries: Dict[str, Any] = {}
    cap = int(max_rows_per_foundation)

    for foundation_idx, foundation in enumerate(foundation_names):
        parts = per_foundation_parts[foundation]
        rows_by_suffix = {
            suffix: int(pool_summaries[suffix]["sources"][foundation]["n_rows"])
            for suffix in suffixes
        }
        if not parts:
            source_summaries[foundation] = {
                "n_rows_before_cap": 0,
                "n_rows": 0,
                "rows_by_suffix": rows_by_suffix,
                "per_class_counts": {},
                "capped_to": cap if cap > 0 else None,
            }
            continue

        X_all = np.concatenate([part[0] for part in parts], axis=0).astype(
            np.float32, copy=False
        )
        y_all = np.concatenate([part[1] for part in parts], axis=0).astype(
            np.int64, copy=False
        )
        n_before_cap = int(len(y_all))

        if cap > 0 and len(y_all) > cap:
            rng = np.random.default_rng(int(seed) + 1009 * foundation_idx)
            keep = rng.choice(len(y_all), size=cap, replace=False)
            keep.sort()
            X_all, y_all = X_all[keep], y_all[keep]

        extras[foundation] = (X_all, y_all)
        source_summaries[foundation] = {
            "n_rows_before_cap": n_before_cap,
            "n_rows": int(len(y_all)),
            "rows_by_suffix": rows_by_suffix,
            "per_class_counts": {
                int(c): int(n) for c, n in zip(*np.unique(y_all, return_counts=True))
            },
            "capped_to": cap if cap > 0 else None,
        }

    return extras, {
        "augmentation_mode": "independent_per_foundation_multi_suffix_train_only_no_parity_required",
        "aug_suffixes": list(suffixes),
        "split": str(split),
        "max_rows_per_foundation": cap if cap > 0 else None,
        "seed": int(seed),
        "search_order": ["completed_parts", "consolidated"],
        "pools": pool_summaries,
        "sources": source_summaries,
    }


# Backward-compatible one-suffix wrapper for callers that imported the old API.
def load_independent_partial_stainaug_extras(
    X_brats: Mapping[str, np.ndarray],
    foundation_names: Sequence[str],
    artifacts_root: Path,
    aug_suffix: str,
    split: str,
    max_rows_per_foundation: int,
    seed: int,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    artifacts_root = Path(artifacts_root)
    return load_independent_stainaug_extras(
        X_brats=X_brats,
        foundation_names=foundation_names,
        artifacts_root=artifacts_root,
        embedding_root=artifacts_root / "embeddings_by_patient_slide",
        aug_suffixes=(aug_suffix,),
        split=split,
        max_rows_per_foundation=max_rows_per_foundation,
        seed=seed,
    )


def _merge_extras(
    *extra_dicts: Mapping[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Concatenate per-source extras across any number of extra sources
    (e.g. Ivy GAP + partial stain-aug). A source missing from one dict simply
    contributes nothing from that dict -- no parity required across dicts
    either, matching the "whatever is available" philosophy throughout."""
    merged: Dict[str, list] = {}
    for d in extra_dicts:
        for source_name, (X, y) in d.items():
            if len(y) == 0:
                continue
            merged.setdefault(source_name, []).append((X, y))
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for source_name, parts in merged.items():
        Xs = [p[0] for p in parts]
        ys = [p[1] for p in parts]
        out[source_name] = (
            np.concatenate(Xs, axis=0).astype(np.float32, copy=False),
            np.concatenate(ys, axis=0).astype(np.int64, copy=False),
        )
    return out


def _write_ivy_train_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_foundation_val(
    root: Path,
    foundation: str,
    val_subpath: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load val .npz for one foundation. Labels may be -1 for the holdout."""
    path = root / foundation / "val" / val_subpath
    if not path.exists():
        raise FileNotFoundError(f"Val file not found: {path}")
    d = np.load(path, allow_pickle=True)
    X = scrub(d["X"].astype(np.float32))
    y = d["y"].astype(np.int64)
    names = d["names"].astype(str)
    return X, y, names


def load_composite_val(
    root: Path,
    foundation_names: Sequence[str],
    val_subpath: str,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load + align val embeddings for all foundations by name intersection."""
    loaded: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for fn in foundation_names:
        loaded[fn] = load_foundation_val(root, fn, val_subpath)
        X, y, names = loaded[fn]
        print(f"  {fn} val: X={X.shape}")

    first = foundation_names[0]
    ref_X, ref_y, ref_names = loaded[first]

    # Fast path: identical order
    if all(np.array_equal(ref_names, loaded[fn][2]) for fn in foundation_names[1:]):
        X_bundle = {fn: loaded[fn][0] for fn in foundation_names}
        return X_bundle, ref_y, ref_names

    # Slow path: align by name intersection
    print("  [align] val patch order differs; aligning by name intersection.")
    member_maps: Dict[str, Dict[str, int]] = {
        fn: {str(n): i for i, n in enumerate(loaded[fn][2])}
        for fn in foundation_names[1:]
    }

    keep_ref_idx: list = []
    aligned_other_idx: Dict[str, list] = {fn: [] for fn in foundation_names[1:]}
    for i, n in enumerate(ref_names):
        key = str(n)
        if all(key in member_maps[fn] for fn in foundation_names[1:]):
            keep_ref_idx.append(i)
            for fn in foundation_names[1:]:
                aligned_other_idx[fn].append(member_maps[fn][key])

    if not keep_ref_idx:
        raise RuntimeError("Composite val: no overlapping patch names found.")

    keep_ref_idx_arr = np.asarray(keep_ref_idx, dtype=np.int64)
    X_bundle: Dict[str, np.ndarray] = {first: ref_X[keep_ref_idx_arr]}
    for fn in foundation_names[1:]:
        idx = np.asarray(aligned_other_idx[fn], dtype=np.int64)
        X_bundle[fn] = loaded[fn][0][idx]

    n_kept = len(keep_ref_idx_arr)
    n_ref = len(ref_names)
    if n_kept < n_ref:
        print(f"  [align] val: kept {n_kept:,}/{n_ref:,} patches.")
    return X_bundle, ref_y[keep_ref_idx_arr], ref_names[keep_ref_idx_arr]


# ── ADDITIVE, opt-in: export fitted heads + manifest.json for Docker ────────
#
# Everything above this point is the original script, unmodified. This
# function (and the --export-docker-ckpts flag in main(), below) is new,
# off by default, and does not change training, TTA, val prediction, or any
# existing output file when not invoked. It operates on the SAME `model`
# object main() already fit -- it does not retrain or refit anything.


def export_docker_checkpoints(
    model: "BratsPath2025ChunkedSGDEnsemble",
    out_dir: Path,
    foundation_dims: Mapping[str, int],
    ivy_train_summary: Mapping[str, Any],
    stainaug_summary: Mapping[str, Any],
    ridge_probe_foundations: Sequence[str],
) -> None:
    """Serialize every fitted chunk head (joblib) plus a manifest.json, laid
    out exactly as the BraTS-Path 2026 Docker submission's
    docker_template/src/ckpts/ expects (manifest.json + heads/...).

    Classifier identity for each entry in model.heads_ is recovered
    positionally: _fit_heads() builds model.heads_ as
    [all primary SGD heads, in chunk order] followed by
    [all probe heads, in chunk order], and model.probe_heads_ records exactly
    the probe heads in that same append order. So heads_[:n_primary] are
    primary SGD heads, and heads_[n_primary:] correspond 1:1, in order, with
    probe_heads_. This is checked explicitly below (not just assumed) --
    if _fit_heads()'s append order ever changes, this raises loudly instead
    of silently mislabelling heads.
    """
    import joblib

    out_dir = Path(out_dir)
    heads_dir = out_dir / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)

    n_probe_heads = len(model.probe_heads_)
    n_primary_heads = len(model.heads_) - n_probe_heads
    if n_primary_heads < 0:
        raise RuntimeError(
            "export_docker_checkpoints: model.probe_heads_ is longer than "
            "model.heads_ -- this shouldn't happen; refusing to export."
        )

    # Fixed BraTS-Path class-id -> short-name mapping, matching the manifest
    # convention already used elsewhere in this project. The training script
    # itself never names classes (only numeric ids 0-9) -- double-check this
    # mapping against your own labelling convention before relying on it.
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

    head_records: List[Dict[str, Any]] = []
    for i, (source_name, start, stop, clf) in enumerate(model.heads_):
        if i < n_primary_heads:
            classifier_name = "sgd_lr_log_a3e-5"
            group = "primary"
        else:
            probe_record = model.probe_heads_[i - n_primary_heads]
            if (
                probe_record["source"],
                probe_record["start"],
                probe_record["stop"],
            ) != (source_name, start, stop):
                raise RuntimeError(
                    f"export_docker_checkpoints: probe_heads_[{i - n_primary_heads}] "
                    f"does not line up positionally with heads_[{i}] "
                    f"({probe_record} vs {(source_name, start, stop)}). "
                    "model.heads_ append order must have changed upstream; update "
                    "the positional split logic above before trusting this export."
                )
            classifier_name = probe_record["probe_name"]
            group = "probe"

        rel_path = (
            Path("heads")
            / source_name
            / classifier_name
            / (f"{source_name}__chunk-{start:05d}-{stop:05d}__{classifier_name}.joblib")
        )
        abs_path = out_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, abs_path)

        head_records.append(
            {
                "foundation": source_name,
                "classifier": classifier_name,
                "group": group,
                "chunk_index": int(start) // int(model.chunk_size),
                "start": int(start),
                "stop": int(stop),
                "dimension": int(stop - start),
                "path": str(rel_path.as_posix()),
                "classes": [int(c) for c in getattr(clf, "classes_", CLASS_IDS)],
            }
        )

    manifest = {
        "schema": "brats_path_2026_docker_ckpts_v1",
        "script_version": SCRIPT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "foundations": list(FOUNDATION_NAMES),
        "dimensions": {fn: int(d) for fn, d in foundation_dims.items()},
        "class_ids": CLASS_IDS.tolist(),
        "class_names": CLASS_NAMES,
        "n_heads": len(head_records),
        "heads": head_records,
        "configuration": {
            "chunk_size": int(model.chunk_size),
            "min_chunk_dim": int(model.min_chunk_dim),
            "sgd": {
                "loss": "log_loss",
                "alpha": 3e-5,
                "penalty": "l2",
                "class_weight": "balanced",
                "max_iter": 20,
                "early_stopping": True,
            },
            "ridge": {
                "alpha": 10.0,
                "solver": "lsqr",
                "class_weight": "balanced",
                "tol": 1e-2,
                "max_iter": 50,
            },
            "source_aggregation": str(model.source_aggregation),
            "rare_boost": float(model.rare_boost),
            "rare_quantile": float(model.rare_quantile),
            "rare_classes": model.rare_classes_.astype(int).tolist(),
            "threshold_grid_size": int(model.threshold_grid_size),
            "threshold_passes": int(model.threshold_passes),
            "thresholds": model.thresholds_.astype(float).tolist(),
            "ridge_probe_foundations": list(ridge_probe_foundations),
        },
        "source_fit_rows": {k: int(v) for k, v in model.source_fit_rows_.items()},
        "source_extra_rows": {k: int(v) for k, v in model.source_extra_rows_.items()},
        "augmentation": {
            "ivy_extras": ivy_train_summary,
            "stainaug_extras": stainaug_summary,
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    print(f"\n[export] Wrote {len(head_records)} head(s) -> {heads_dir}")
    print(f"[export] Wrote manifest -> {manifest_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    ap.add_argument(
        "--embedding-root",
        default="artifacts/embeddings_by_patient_slide",
        help="Root directory containing per-foundation consolidated embeddings.",
    )
    ap.add_argument(
        "--val-subpath",
        default="patient-unmapped/slide-unmapped.npz",
        help="Path inside {embedding_root}/{foundation}/val/ to the val .npz file.",
    )
    ap.add_argument(
        "--out-csv",
        default="chunksgd_c768_source_allf_ridgeprobe_predictions.csv",
        help="Output submission CSV: SubjectID, Prediction.",
    )
    ap.add_argument(
        "--out-proba",
        default="artifacts/final_predictions/chunksgd_c768_source_allf_ridgeprobe_proba.npz",
        help="Output NPZ: names, proba [N,10] float16, pred [N] int16.",
    )
    # ── Independent labelled Ivy GAP training augmentation ─────────────────────
    ap.add_argument(
        "--ivy-virchow2-root",
        default="/Users/agata/competitions/brats/ivygap_embeddings_raw/virchow2",
        help="Root containing raw Ivy Virchow2 embedding_config.json and blocks/.",
    )
    ap.add_argument(
        "--ivy-hoptimus1-root",
        default="/Users/agata/competitions/brats/ivygap_embeddings_raw/hoptimus1",
        help="Root containing raw Ivy H-optimus-1 embedding_config.json and blocks/.",
    )
    ap.add_argument(
        "--ivy-genbiopathfm-root",
        default="/Users/agata/competitions/brats/ivygap_embeddings_raw/genbiopathfm",
        help="Root containing raw Ivy GenBioPathFM embedding_config.json and blocks/.",
    )
    ap.add_argument(
        "--ivy-max-per-class",
        type=int,
        default=8120,
        help="Independent Ivy rows retained per supported class for EACH of Virchow2, "
        "H-optimus-1, and GenBioPathFM. No cross-foundation UID intersection. 0 = every "
        "available row per supported class (CT/IC/MP/NC/PN).",
    )
    ap.add_argument(
        "--ivy-seed",
        type=int,
        default=SEED,
        help="Deterministic independent-Ivy sampling seed (source-specific offsets are used).",
    )
    ap.add_argument(
        "--ivy-norm-mode",
        choices=("auto", "l2", "none"),
        default="auto",
        help="How raw Ivy features are matched to existing BraTS source features. "
        "auto detects unit-norm BraTS sources and L2-normalizes only Ivy rows "
        "in RAM; l2 forces that action; none leaves Ivy raw.",
    )
    ap.add_argument(
        "--disable-ivy-extras",
        action="store_true",
        help="Skip loading Ivy GAP extras entirely (e.g. if the raw Ivy roots aren't available).",
    )
    # ── Independent partial stain-augmentation training extras ─────────────────
    ap.add_argument(
        "--stainaug-artifacts-root",
        default="artifacts",
        help="Artifacts root matching --artifacts in extract_augmented_embeddings-2.py "
        "(that script writes under {this}/embedding_parts/<foundation>_<suffix>/<split>/).",
    )
    ap.add_argument(
        "--stainaug-suffixes",
        nargs="+",
        default=("stainaug", "stainaug_local"),
        help="One or more augmentation suffixes. By default both training pools are used: "
        "<foundation>_stainaug and <foundation>_stainaug_local. Comma-separated "
        "values are also accepted.",
    )
    ap.add_argument(
        "--stainaug-suffix",
        action="append",
        dest="legacy_stainaug_suffixes",
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--stainaug-split",
        default="train",
        help="Split subdirectory to read partial stain-aug shards from.",
    )
    ap.add_argument(
        "--stainaug-max-rows-per-foundation",
        type=int,
        default=0,
        help="Cap on partial stain-aug rows used per foundation (uniform random subsample "
        "if exceeded). 0 = use everything completed so far for that foundation, "
        "however much or little that is.",
    )
    ap.add_argument(
        "--stainaug-seed",
        type=int,
        default=SEED,
        help="Seed used only if --stainaug-max-rows-per-foundation triggers a subsample.",
    )
    ap.add_argument(
        "--disable-stainaug-extras",
        action="store_true",
        help="Skip loading partial stain-augmentation extras entirely.",
    )
    # ── Additional probe estimator (Ridge, now generalized to any/all foundations) ──
    ap.add_argument(
        "--ridge-probe-foundations",
        nargs="+",
        default=list(FOUNDATION_NAMES),
        help="Which foundations get the extra ridge_lsqr_a10_l2_balanced probe heads. "
        f"Choose any subset of {list(FOUNDATION_NAMES)}. Comma-separated values are "
        "also accepted (e.g. 'virchow2,hoptimus1'). Defaults to ALL foundations "
        "(this used to be virchow2-only). Pass an empty selection or use "
        "--disable-ridge-probe to add no probes at all.",
    )
    ap.add_argument(
        "--disable-ridge-probe",
        action="store_true",
        help="Skip adding the extra ridge_lsqr_a10_l2_balanced probe heads entirely "
        "(overrides --ridge-probe-foundations) and fall back to plain "
        "chunk-SGD-only heads for every source.",
    )
    # ── TTA (test-time chunked feature masking) ───────────────────────────────
    ap.add_argument(
        "--tta-aug",
        type=int,
        default=0,
        help="Number of masked TTA passes. 0 = identical to base config (no TTA).",
    )
    ap.add_argument(
        "--tta-keep",
        type=float,
        default=0.9,
        help="Per-dimension keep probability for masking (mask fraction = 1 - keep). "
        "Kept dims are rescaled by 1/keep (inverted dropout).",
    )
    ap.add_argument(
        "--tta-seed",
        type=int,
        default=SEED,
        help="RNG seed for the TTA masks (predictions are reproducible).",
    )
    # ── ADDITIVE, opt-in: export fitted heads + manifest for Docker packaging ──
    # Everything above this point is 100% unchanged from the original script.
    # This flag is off by default and changes NOTHING about training, TTA, val
    # prediction, or any existing output file when omitted.
    ap.add_argument(
        "--export-docker-ckpts",
        default=None,
        help="If set, after fitting, additionally export every fitted chunk head "
        "(joblib) plus a manifest.json compatible with the BraTS-Path 2026 "
        "Docker submission's src/ckpts/ layout, to this directory. Does not "
        "affect training, val prediction, or any other existing output.",
    )
    args = ap.parse_args()

    raw_stainaug_suffixes = (
        args.legacy_stainaug_suffixes
        if args.legacy_stainaug_suffixes is not None
        else args.stainaug_suffixes
    )
    stainaug_suffixes = _normalize_stainaug_suffixes(raw_stainaug_suffixes)

    # Normalize the ridge-probe foundation selection (supports comma-separated
    # tokens the same way --stainaug-suffixes does), validated against the
    # known foundation names.
    ridge_probe_foundations: Tuple[str, ...] = tuple()
    if not args.disable_ridge_probe:
        requested: List[str] = []
        for value in args.ridge_probe_foundations:
            for token in str(value).split(","):
                token = token.strip()
                if token and token not in requested:
                    requested.append(token)
        unknown_foundations = sorted(set(requested) - set(FOUNDATION_NAMES))
        if unknown_foundations:
            raise SystemExit(
                f"--ridge-probe-foundations: unknown foundation(s) {unknown_foundations}; "
                f"expected a subset of {list(FOUNDATION_NAMES)}."
            )
        # Preserve FOUNDATION_NAMES order for stable, deterministic head ordering.
        ridge_probe_foundations = tuple(
            fn for fn in FOUNDATION_NAMES if fn in requested
        )

    embedding_root = Path(args.embedding_root)
    out_proba = Path(args.out_proba)
    out_proba.parent.mkdir(parents=True, exist_ok=True)

    # ── Load composite train ──────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Loading composite TRAIN  foundations={list(FOUNDATION_NAMES)}")
    print(f"  root: {embedding_root}")
    print(f"{'─' * 60}")
    X_train_bundle, y_train, train_names, train_patients = load_composite_train(
        embedding_root,
        FOUNDATION_NAMES,
    )
    total_patches = int(next(iter(X_train_bundle.values())).shape[0])
    print(f"\n  Total aligned BraTS train patches : {total_patches:,}")
    print(
        f"  BraTS class distribution          : {np.bincount(y_train, minlength=N_CLASSES).tolist()}"
    )
    for fn, Xf in X_train_bundle.items():
        print(f"  {fn} dim = {Xf.shape[1]}")

    # ── Independent Ivy augmentation for source-specific head fitting ─────────
    ivy_source_extras: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    ivy_train_summary: Dict[str, Any] = {"skipped": True}
    if not args.disable_ivy_extras:
        print(f"\n{'─' * 60}")
        print(
            "Loading independent labelled Ivy GAP extras for source-specific TRAIN augmentation"
        )
        print(
            f"  virchow2 root : {Path(args.ivy_virchow2_root).expanduser().resolve()}"
        )
        print(
            f"  hoptimus1 root: {Path(args.ivy_hoptimus1_root).expanduser().resolve()}"
        )
        print(
            f"  genbiopathfm root: {Path(args.ivy_genbiopathfm_root).expanduser().resolve()}"
        )
        print(
            f"  cap/class/source: {args.ivy_max_per_class}  (0 means all rows per source)"
        )
        print("  pairing       : disabled (no cross-foundation patch_uid intersection)")
        print(f"  norm mode     : {args.ivy_norm_mode}")
        print(f"{'─' * 60}")
        ivy_source_extras, ivy_train_summary = load_independent_ivy_source_extras(
            X_brats=X_train_bundle,
            virchow2_root=Path(args.ivy_virchow2_root),
            hoptimus1_root=Path(args.ivy_hoptimus1_root),
            genbiopathfm_root=Path(args.ivy_genbiopathfm_root),
            max_per_class=args.ivy_max_per_class,
            seed=args.ivy_seed,
            norm_mode=args.ivy_norm_mode,
        )
        for fn in FOUNDATION_NAMES:
            source = ivy_train_summary["sources"][fn]
            selection = source["selection"]
            action = source["representation_matching"]["action"]
            print(f"\n  {fn} independent Ivy rows      : {selection['n_selected']:,}")
            print(
                f"  {fn} selected by class          : {selection['selected_by_class']}"
            )
            print(f"  {fn} representation action      : {action}")
    else:
        print("\n[skip] --disable-ivy-extras set; no Ivy GAP extras will be used.")

    # ── Independent multi-pool stain augmentation (asymmetric OK) ─────────────
    stainaug_source_extras: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    stainaug_summary: Dict[str, Any] = {"skipped": True}
    if not args.disable_stainaug_extras:
        print(f"\n{'─' * 60}")
        print("Loading independent stain-augmentation TRAIN pools")
        print(
            f"  artifacts root : {Path(args.stainaug_artifacts_root).expanduser().resolve()}"
        )
        print(f"  embedding root : {embedding_root.expanduser().resolve()}")
        print(f"  suffixes       : {list(stainaug_suffixes)}")
        print(
            f"  foundations    : {list(FOUNDATION_NAMES)}  (independent; no parity required)"
        )
        print("  search order   : completed shard parts, then consolidated embeddings")
        print(f"{'─' * 60}")
        stainaug_source_extras, stainaug_summary = load_independent_stainaug_extras(
            X_brats=X_train_bundle,
            foundation_names=FOUNDATION_NAMES,
            artifacts_root=Path(args.stainaug_artifacts_root),
            embedding_root=embedding_root,
            aug_suffixes=stainaug_suffixes,
            split=args.stainaug_split,
            max_rows_per_foundation=args.stainaug_max_rows_per_foundation,
            seed=args.stainaug_seed,
        )

        for suffix in stainaug_suffixes:
            print(f"\n  pool={suffix}")
            for fn in FOUNDATION_NAMES:
                s = stainaug_summary["pools"][suffix]["sources"][fn]
                if not s.get("found", False):
                    print(
                        f"    {fn}: 0 rows (no completed parts or consolidated files)"
                    )
                    continue
                print(
                    f"    {fn}: {s['n_rows']:,} rows from {s['n_slide_files']:,} slide files "
                    f"[{s['source_kind']}]"
                )
                print(f"      classes: {s['per_class_counts']}")

        print("\n  merged stain-augmentation pool per foundation")
        for fn in FOUNDATION_NAMES:
            s = stainaug_summary["sources"][fn]
            print(
                f"    {fn}: {s['n_rows']:,} rows "
                f"(before cap={s['n_rows_before_cap']:,}; by suffix={s['rows_by_suffix']})"
            )
            if s["n_rows"]:
                print(f"      classes after merge/cap: {s['per_class_counts']}")
    else:
        print(
            "\n[skip] --disable-stainaug-extras set; no stain-augmentation extras will be used."
        )

    # ── Merge extras from both independent-augmentation sources per foundation ─
    combined_source_extras = _merge_extras(ivy_source_extras, stainaug_source_extras)
    print(f"\n{'─' * 60}")
    print("Combined per-source training extras (Ivy GAP + stainaug + stainaug_local):")
    for fn in FOUNDATION_NAMES:
        n = (
            int(len(combined_source_extras[fn][1]))
            if fn in combined_source_extras
            else 0
        )
        n_ivy = int(len(ivy_source_extras[fn][1])) if fn in ivy_source_extras else 0
        n_aug = (
            int(len(stainaug_source_extras[fn][1]))
            if fn in stainaug_source_extras
            else 0
        )
        by_suffix = (
            stainaug_summary.get("sources", {}).get(fn, {}).get("rows_by_suffix", {})
        )
        print(
            f"  {fn}: total extra rows = {n:,}  (ivy={n_ivy:,} + stainaug_pools={n_aug:,}; by_suffix={by_suffix})"
        )
    print(f"{'─' * 60}")

    print(f"\n  Aligned BraTS train rows remain : {total_patches:,}")
    print(
        f"  BraTS class distribution remains : {np.bincount(y_train, minlength=N_CLASSES).tolist()}"
    )

    # ── Load composite val ────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Loading composite VAL  foundations={list(FOUNDATION_NAMES)}")
    print(f"{'─' * 60}")
    X_val_bundle, y_val, val_names = load_composite_val(
        embedding_root,
        FOUNDATION_NAMES,
        args.val_subpath,
    )
    n_val = int(next(iter(X_val_bundle.values())).shape[0])
    print(f"\n  Total aligned val patches : {n_val:,}")
    has_labels = bool(n_val > 0 and (y_val >= 0).all())
    if (y_val >= 0).any():
        uv, cv = np.unique(y_val[y_val >= 0], return_counts=True)
        print(
            f"  Val class distribution    : { {int(k): int(v) for k, v in zip(uv, cv)} }"
        )
    else:
        print("  Labels: all -1 (unlabelled — predictions only, no metrics)")

    # ── Additional probe estimator config: Ridge, generalized to any/all sources ──
    # ProbeConfig(
    #     "ridge_lsqr_a10_l2_balanced",
    #     RidgeClassifier(alpha=10.0, class_weight="balanced", solver="lsqr", tol=1e-2, max_iter=50),
    #     "l2",
    #     only_foundations=<ridge_probe_foundations>,   # now configurable; defaults to ALL
    #     aug_artifact_suffixes=("ivy", "stainaug", "stainaug_local"),
    # )
    # Each selected foundation's probe is fit on exactly that foundation's own
    # fit rows as assembled above (base BraTS rows + the already-merged Ivy GAP
    # + stainaug + stainaug_local extras) -- no separate extras loading is
    # needed, and each foundation gets its own independently-fit Ridge instance.
    probe_estimators: Dict[str, List[Tuple[str, Any]]] = {}
    if ridge_probe_foundations:
        for fn in ridge_probe_foundations:
            probe_estimators[fn] = [
                ("ridge_lsqr_a10_l2_balanced", _ridge_lsqr_a10_l2_balanced()),
            ]
        print(f"\n{'─' * 60}")
        print("Additional probe estimator enabled")
        print(
            "  probe   : ridge_lsqr_a10_l2_balanced "
            "(RidgeClassifier alpha=10.0, solver=lsqr, class_weight=balanced)"
        )
        print(f"  scope   : {list(ridge_probe_foundations)}")
        print(
            "  fit data: each foundation's own fit rows "
            "(BraTS base + ivy + stainaug + stainaug_local extras, same as SGD heads)"
        )
        print(f"{'─' * 60}")
    else:
        print(
            "\n[skip] no ridge-probe foundations selected "
            "(--disable-ridge-probe or empty --ridge-probe-foundations); no probe heads will be added."
        )

    # ── Build model: sweep's chunksgd_a3e-5_c768_source config, plus the ─────
    #    optional Ridge probe(s) layered into the selected source bucket(s).
    model = BratsPath2025ChunkedSGDEnsemble(
        foundation_names=FOUNDATION_NAMES,
        foundation_label="composite__virchow2+hoptimus1+genbiopathfm__ivy_plus_stainaug_plus_stainaug_local_train_aug"
        "__plus_ridge_lsqr_a10_probe_on_"
        + ("+".join(ridge_probe_foundations) if ridge_probe_foundations else "none"),
        base_estimator=_sgd(alpha=3e-5, loss="log_loss"),
        chunk_size=768,
        min_chunk_dim=16,
        rare_boost=1.10,
        rare_quantile=0.35,
        rare_classes=None,
        calibration_fraction=0.10,
        threshold_grid_size=11,
        threshold_passes=1,
        max_train_samples_per_class=None,
        use_sample_weight=False,
        source_weights=None,
        source_aggregation="source_mean",  # matches the *_source config
        probe_estimators=probe_estimators,
        verbose=-1,
        random_state=SEED,
    )
    n_chunks_per_foundation = {
        fn: sum(1 for _ in model._iter_chunks(X_train_bundle[fn].shape[1]))
        for fn in FOUNDATION_NAMES
    }
    n_probe_chunks = {
        source_name: {
            probe_name: sum(
                1 for _ in model._iter_chunks(X_train_bundle[source_name].shape[1])
            )
            for probe_name, _ in probes
        }
        for source_name, probes in probe_estimators.items()
    }
    total_heads = sum(n_chunks_per_foundation.values()) + sum(
        sum(v.values()) for v in n_probe_chunks.values()
    )
    print(f"\n{'─' * 60}")
    print("Training  chunksgd_a3e-5_c768_source  (source_mean aggregation)")
    print(f"  chunk_size      : {model.chunk_size}")
    print(
        f"  SGD heads/source: { {fn: n for fn, n in n_chunks_per_foundation.items()} }"
    )
    print(f"  probe heads     : {n_probe_chunks}")
    print(f"  total heads     : {total_heads}")
    print(f"  source_agg      : {model.source_aggregation}")
    print(f"  rare_boost      : {model.rare_boost}  quantile={model.rare_quantile}")
    print(f"  cal_fraction    : {model.calibration_fraction}")
    print(f"{'─' * 60}")

    t0 = time.time()
    model.fit(
        X_train_bundle,
        y_train,
        source_train_extras=combined_source_extras,
    )  # BraTS-only calibration; Ivy/stainaug/stainaug_local affect source-head fitting only
    elapsed = time.time() - t0
    print(f"\n  Training finished in {elapsed:.1f}s")
    print(f"  Rare classes identified: {model.rare_classes_.tolist()}")
    print(f"  Calibrated thresholds  : {model.thresholds_.round(4).tolist()}")
    print(f"  Rows actually used per source (base + extras): {model.source_fit_rows_}")
    print(
        f"  Extra rows contributed per source            : {model.source_extra_rows_}"
    )
    print(
        f"  Probe heads fitted                            : {len(model.probe_heads_)}"
    )
    if model.probe_heads_:
        by_probe: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        for h in model.probe_heads_:
            by_probe[h["probe_name"]] = by_probe.get(h["probe_name"], 0) + 1
            by_source[h["source"]] = by_source.get(h["source"], 0) + 1
        print(f"    breakdown by probe name                     : {by_probe}")
        print(f"    breakdown by source                         : {by_source}")

    # ── ADDITIVE, opt-in: export fitted heads + manifest for Docker packaging ──
    # Uses the SAME `model` object that was just fit above -- not a re-fit, not
    # a separate training path. Off by default; only runs if you pass
    # --export-docker-ckpts. See export_docker_checkpoints() for details.
    if args.export_docker_ckpts:
        export_docker_checkpoints(
            model=model,
            out_dir=Path(args.export_docker_ckpts),
            foundation_dims={
                fn: int(X_train_bundle[fn].shape[1]) for fn in FOUNDATION_NAMES
            },
            ivy_train_summary=ivy_train_summary,
            stainaug_summary=stainaug_summary,
            ridge_probe_foundations=ridge_probe_foundations,
        )

    del X_train_bundle, y_train, train_names, train_patients
    gc.collect()

    # ── Predict on val (optionally with chunked feature-masking TTA) ───────────
    print(f"\n{'─' * 60}")
    print("Predicting on val ...")
    if int(args.tta_aug) > 0:
        print(
            f"  TTA: {args.tta_aug} masked pass(es) + 1 clean pass  "
            f"(averaged as RAW posteriors)"
        )
        print(
            f"       keep={args.tta_keep:.3f}  mask_frac={1.0 - args.tta_keep:.3f}  "
            f"seed={args.tta_seed}"
        )
        print("       rare-boost + thresholds applied ONCE after averaging")
    else:
        print(
            "  TTA disabled (--tta-aug 0) → identical to chunksgd_a3e-5_c768_source"
            " (+ ridge probe(s), if enabled)"
        )
    print(f"{'─' * 60}")

    proba = model.predict_proba_tta(
        X_val_bundle,
        tta_aug=args.tta_aug,
        tta_keep=args.tta_keep,
        tta_seed=args.tta_seed,
    )
    pred = proba.argmax(axis=1).astype(np.int16)

    # Clean baseline for an in-run comparison (only meaningful when TTA is on).
    proba_clean = None
    if int(args.tta_aug) > 0:
        proba_clean = model.predict_proba(X_val_bundle)
        pred_clean = proba_clean.argmax(axis=1).astype(np.int16)
        agree = float((pred == pred_clean).mean()) * 100.0
        print(
            f"\n  clean vs TTA argmax agreement: {agree:.2f}%  "
            f"({int((pred != pred_clean).sum()):,} of {n_val:,} flipped)"
        )

    pred_unique, pred_counts = np.unique(pred, return_counts=True)
    print(
        "\n  Predicted class distribution (TTA output):"
        if int(args.tta_aug) > 0
        else "\n  Predicted class distribution:"
    )
    for cls, cnt in zip(pred_unique, pred_counts):
        print(
            f"    class {int(cls):2d}: {cnt:>8,}  ({100.0 * cnt / max(n_val, 1):.1f}%)"
        )

    # ── Metrics (only when val labels are available) ──────────────────────────
    if has_labels:

        def _report(tag: str, pr: np.ndarray) -> None:
            mcc = matthews_corrcoef(y_val, pr)
            f1 = f1_score(y_val, pr, average="macro", zero_division=0)
            acc = accuracy_score(y_val, pr)
            print(
                f"    [{tag:<5}] MCC={mcc:.4f}  macro-F1={f1:.4f}  accuracy={acc:.4f}"
            )

        print(f"\n  Val metrics:")
        if proba_clean is not None:
            _report("clean", proba_clean.argmax(axis=1))
            _report("tta", pred)
        else:
            _report("base", pred)

    # ── Save outputs (the TTA result is the headline submission) ──────────────
    out_df = pd.DataFrame(
        {
            "SubjectID": val_names.astype(str),
            "Prediction": pred.astype(int),
        }
    )
    out_df.to_csv(args.out_csv, index=False)

    np.savez(
        out_proba,
        names=val_names.astype(object),
        proba=proba.astype(np.float16),
        pred=pred,
    )

    ivy_summary_path = out_proba.with_name(
        out_proba.stem + "_train_augmentation_summary.json"
    )
    combined_summary = {
        "script_version": SCRIPT_VERSION,
        "aligned_brats_train_rows": int(total_patches),
        "source_head_fit_rows": {k: int(v) for k, v in model.source_fit_rows_.items()},
        "source_extra_rows_total": {
            k: int(v) for k, v in model.source_extra_rows_.items()
        },
        "foundation_names": list(FOUNDATION_NAMES),
        "ivy_extras": ivy_train_summary,
        "stainaug_extras": stainaug_summary,
        "partial_stainaug_extras": stainaug_summary,  # compatibility with v1 summaries
        "ridge_probe_foundations": list(ridge_probe_foundations),
        "probe_estimators": {
            source_name: [name for name, _ in probes]
            for source_name, probes in probe_estimators.items()
        },
        "probe_heads_fitted": model.probe_heads_,
        "model_config_unchanged": {
            "chunk_size": int(model.chunk_size),
            "source_aggregation": str(model.source_aggregation),
            "rare_boost": float(model.rare_boost),
            "rare_quantile": float(model.rare_quantile),
            "calibration_fraction": float(model.calibration_fraction),
            "threshold_grid_size": int(model.threshold_grid_size),
            "threshold_passes": int(model.threshold_passes),
            "tta_aug": int(args.tta_aug),
            "tta_keep": float(args.tta_keep),
        },
        "rare_classes_identified_after_augmentation": model.rare_classes_.astype(
            int
        ).tolist(),
        "calibrated_thresholds_after_augmentation": model.thresholds_.astype(
            float
        ).tolist(),
    }
    _write_ivy_train_summary(ivy_summary_path, combined_summary)

    print(f"\nWrote {args.out_csv}: {len(out_df):,} rows")
    print(f"Wrote {out_proba}")
    print(f"Wrote {ivy_summary_path}")
    print(out_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
