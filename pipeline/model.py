"""Rank-blended ensemble for Poker44 chunk classification. -- v6 RANKING-OBJECTIVE ARM.

v6 differs from v3 in ONE thing: the middle member no longer learns a per-chunk
classification objective, it learns a LISTWISE RANKING objective (LambdaRank).

Why: the reward gate is `true_positives == 0` -> reward 0, and we serve by flagging
the top pos_frac of chunks BY RANK (pipeline/threshold.shape_gate_safe). So the only
thing that can save us is whether a real bot lands in the top-k of a 100-chunk
request. That is a listwise top-k problem, not a per-chunk probability problem.
A classifier spends capacity being right about the easy 84 chunks it will never flag;
LambdaRank spends it on the head of the ranking, which is the only region we serve.

Members (fused by in-batch rank, calibration-agnostic):
  A. stacked GBDT: LightGBM + XGBoost + CatBoost + ExtraTrees -> LogisticRegression
  B. LambdaRank LightGBM trio (listwise, grouped per date, truncated to the head)
  C. PCA -> MLP trio (different model family for decorrelation)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, LGBMRanker
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def mine_monotone_signs(
    X: np.ndarray,
    y: np.ndarray,
    dates: List[str],
    *,
    min_abs_rho: float = 0.15,
) -> List[int]:
    """Per-feature sign constraints that hold across every training date."""
    unique_dates = sorted(set(dates))
    dates_arr = np.asarray(dates)
    signs = np.zeros(X.shape[1], dtype=int)
    for j in range(X.shape[1]):
        rhos = []
        for d in unique_dates:
            mask = dates_arr == d
            if mask.sum() < 8 or len(set(y[mask])) < 2:
                continue
            rho = spearmanr(X[mask, j], y[mask]).statistic
            if np.isfinite(rho):
                rhos.append(rho)
        if len(rhos) >= 3:
            arr = np.asarray(rhos)
            if abs(arr.mean()) >= min_abs_rho and (np.sign(arr) == np.sign(arr.mean())).mean() >= 0.8:
                signs[j] = int(np.sign(arr.mean()))
    return signs.tolist()


def build_stack(seed: int = 0) -> StackingClassifier:
    return StackingClassifier(
        estimators=[
            ("lgbm", LGBMClassifier(
                n_estimators=500, num_leaves=96, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7, min_child_samples=10,
                random_state=seed, verbose=-1, n_jobs=4)),
            ("xgb", XGBClassifier(
                n_estimators=500, max_leaves=64, grow_policy="lossguide",
                learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                random_state=seed, n_jobs=4, eval_metric="logloss")),
            ("cat", CatBoostClassifier(
                iterations=600, depth=7, learning_rate=0.04,
                random_seed=seed, verbose=0, thread_count=4)),
            ("et", ExtraTreesClassifier(
                n_estimators=400, max_depth=18, min_samples_leaf=3,
                random_state=seed, n_jobs=4)),
        ],
        final_estimator=LogisticRegression(C=0.5, max_iter=2000),
        stack_method="predict_proba",
        cv=4,
        n_jobs=1,
    )


def build_mono(signs: List[int]) -> VotingClassifier:
    members = [
        (f"mlgb{s}", LGBMClassifier(
            n_estimators=450, num_leaves=63, learning_rate=0.035,
            subsample=0.85, colsample_bytree=0.8, min_child_samples=12,
            monotone_constraints=signs, random_state=100 + s,
            verbose=-1, n_jobs=4))
        for s in range(3)
    ]
    return VotingClassifier(members, voting="soft", n_jobs=1)


# -- v6: listwise ranking member -------------------------------------------
# TRUNCATION is the head we actually serve. We flag the top pos_frac (0.16) of a
# ~100-chunk request, so ordering beyond the top ~20 is never acted on; telling
# LambdaRank to stop caring past there concentrates capacity where the gate is won.
RANK_TRUNCATION = 20


def build_rankers(seed_base: int = 300) -> List[LGBMRanker]:
    """Trio of LambdaRank LightGBMs (seed-diverse, averaged like the other members)."""
    return [
        LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            lambdarank_truncation_level=RANK_TRUNCATION,
            label_gain=[0, 1],  # binary relevance: human=0, bot=1
            n_estimators=450, num_leaves=63, learning_rate=0.035,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.8,
            min_child_samples=12, random_state=seed_base + s,
            verbose=-1, n_jobs=4,
        )
        for s in range(3)
    ]


def build_mlp() -> VotingClassifier:
    members = [
        (f"mlp{s}", Pipeline([
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=56, random_state=200 + s)),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(80,), alpha=1e-3, max_iter=600,
                early_stopping=True, random_state=200 + s)),
        ]))
        for s in range(3)
    ]
    return VotingClassifier(members, voting="soft", n_jobs=3)


class RankBlend:
    """Weighted rank fusion of the three members.

    v6: the "mono" classifier member is replaced by "ranker" (LambdaRank trio) and
    given the dominant weight. Deliberately a big swing, not a nudge -- the live
    reward is a 1-bit readout (top-10 or 0.00), so a variant that only nudges the
    ordering would be indistinguishable from v3 in the only signal we can read.
    """

    WEIGHTS = {"stack": 0.25, "ranker": 0.50, "mlp": 0.25}

    def __init__(self):
        self.stack: Optional[StackingClassifier] = None
        self.rankers: Optional[List[LGBMRanker]] = None
        self.mlp: Optional[VotingClassifier] = None
        self.cols: List[str] = []
        self.meta: Dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray, dates: List[str], cols: List[str]):
        self.cols = list(cols)
        # LambdaRank needs one "query" per scoring context. A validator request is
        # ~100 chunks drawn from one period, and train.py/retrain.py already rank
        # within a date -- so a date IS the query group. LightGBM requires group
        # members contiguous, hence the stable sort.
        order = np.argsort(np.asarray(dates), kind="mergesort")
        Xg, yg = X[order], np.asarray(y)[order]
        _, counts = np.unique(np.asarray(dates)[order], return_counts=True)
        self.meta["n_groups"] = int(len(counts))
        self.meta["group_sizes"] = [int(c) for c in counts[:5]]

        self.stack = build_stack().fit(X, y)
        self.rankers = [r.fit(Xg, yg, group=counts) for r in build_rankers()]
        self.mlp = build_mlp().fit(X, y)
        return self

    @staticmethod
    def _squash(s: np.ndarray) -> np.ndarray:
        """LambdaRank emits unbounded scores; the blend averages members on a
        common scale. Sigmoid is monotone, so it changes no ordering -- it only
        stops the ranker's raw magnitude from swamping the 0-1 members."""
        return 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))

    @staticmethod
    def _rank01(p: np.ndarray) -> np.ndarray:
        if len(p) <= 1:
            return np.full_like(p, 0.5, dtype=float)
        order = np.argsort(np.argsort(p, kind="mergesort"))
        return order / (len(p) - 1)

    def member_probs(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "stack": self.stack.predict_proba(X)[:, 1],
            "ranker": self._squash(
                np.mean([r.predict(X) for r in self.rankers], axis=0)
            ),
            "mlp": self.mlp.predict_proba(X)[:, 1],
        }

    def score(self, X: np.ndarray) -> np.ndarray:
        probs = self.member_probs(X)
        w = self.WEIGHTS
        num = sum(w[k] * self._rank01(p) for k, p in probs.items())
        return num / sum(w.values())

    def score_prob(self, X: np.ndarray) -> np.ndarray:
        """Probability-scale blend (batch-size independent, used for serving
        alongside rank fusion)."""
        probs = self.member_probs(X)
        w = self.WEIGHTS
        return sum(w[k] * p for k, p in probs.items()) / sum(w.values())
