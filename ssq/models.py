"""号码概率模型。

每个模型: predict(t) -> (p_red[33], p_blue[16])
  只使用第 t 期之前的数据; p_red 之和为 6 (每期开 6 个红球), p_blue 之和为 1。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT_DATA = "data/draws.csv"
NR, NB = 33, 16
WINDOWS = (5, 10, 20, 50, 100)


def load(path: str = ROOT_DATA) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"issue": str})


def matrices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    reds = df[[f"r{i}" for i in range(1, 7)]].to_numpy() - 1
    R = np.zeros((len(df), NR), dtype=np.int8)
    np.put_along_axis(R, reds, 1, axis=1)
    B = np.zeros((len(df), NB), dtype=np.int8)
    B[np.arange(len(df)), df["blue"].to_numpy() - 1] = 1
    return R, B


def features(M: np.ndarray) -> np.ndarray:
    """F[t, n, k]: 只用 M[:t] 计算的特征 (t = 0..T, 共 T+1 行, 最后一行用于预测下一期)。"""
    T, K = M.shape
    C = np.vstack([np.zeros((1, K)), np.cumsum(M, axis=0)])  # C[t] = sum M[:t]
    feats = []
    for w in WINDOWS:
        lo = np.maximum(np.arange(T + 1) - w, 0)
        span = np.maximum(np.arange(T + 1) - lo, 1)[:, None]
        feats.append((C - C[lo]) / span)
    feats.append(C / np.maximum(np.arange(T + 1), 1)[:, None])  # 全历史频率
    gap = np.zeros((T + 1, K))
    g = np.full(K, 50.0)
    for t in range(T + 1):
        gap[t] = g
        if t < T:
            g = np.where(M[t] == 1, 1.0, g + 1)
    feats.append(np.log1p(np.minimum(gap, 200)))
    last = np.vstack([np.zeros((1, K)), M]).astype(float)
    feats.append(last)
    num = np.broadcast_to(np.arange(1, K + 1) / K, (T + 1, K))
    feats.append(num)
    return np.stack(feats, axis=2)


def _norm(p: np.ndarray, total: float) -> np.ndarray:
    p = np.clip(p, 1e-6, None)
    for _ in range(5):
        p = p * total / p.sum()
        p = np.clip(p, 1e-6, 0.999)
    return p


class Uniform:
    name, label = "uniform", "均匀随机 (基线)"

    def fit(self, *a): return self

    def predict(self, t):
        return np.full(NR, 6 / NR), np.full(NB, 1 / NB)


class Frequency:
    """窗口频率 + 向均匀收缩 (热号假设)。window=None 为全历史。"""

    def __init__(self, window=None, prior=30):
        self.window, self.prior = window, prior
        self.name = f"freq_{window or 'all'}"
        self.label = f"热号频率 ({'全历史' if window is None else f'近{window}期'})"

    def fit(self, R, B):
        self.CR = np.vstack([np.zeros((1, NR)), np.cumsum(R, 0)])
        self.CB = np.vstack([np.zeros((1, NB)), np.cumsum(B, 0)])
        return self

    def predict(self, t):
        lo = 0 if self.window is None else max(0, t - self.window)
        n = t - lo
        cr = self.CR[t] - self.CR[lo]
        cb = self.CB[t] - self.CB[lo]
        pr = (cr + self.prior * 6 / NR) / (n + self.prior)
        pb = (cb + self.prior / NB) / (n + self.prior)
        return _norm(pr, 6), pb / pb.sum()


class Cold:
    """遗漏值 (冷号回补假设): 遗漏越久, 概率越高。"""
    name, label = "cold_gap", "冷号回补 (遗漏值)"

    def __init__(self, strength=0.15):
        self.s = strength

    def fit(self, R, B):
        self.FR, self.FB = features(R), features(B)
        return self

    def predict(self, t):
        gi = len(WINDOWS) + 1
        gr, gb = self.FR[t, :, gi], self.FB[t, :, gi]
        pr = np.exp(self.s * (gr - gr.mean()))
        pb = np.exp(self.s * (gb - gb.mean()))
        return _norm(pr / pr.sum() * 6, 6), pb / pb.sum()


class ML:
    """逐号码二分类 (特征: 多窗口频率/遗漏/上期/号码本身), 滚动重训。"""

    def __init__(self, kind="logreg", retrain=100, min_train=150):
        self.kind, self.retrain, self.min_train = kind, retrain, min_train
        self.name = f"ml_{kind}"
        self.label = {"logreg": "机器学习: 逻辑回归", "gbm": "机器学习: 梯度提升树"}[kind]

    def _model(self):
        if self.kind == "logreg":
            return LogisticRegression(max_iter=500, C=0.1)
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05,
                                              max_leaf_nodes=15, l2_regularization=1.0)

    def fit(self, R, B):
        self.R, self.B = R, B
        self.FR, self.FB = features(R), features(B)
        self.cache = {}
        return self

    def _train(self, F, M, t0):
        ts = np.arange(self.min_train, t0)
        X = F[ts].reshape(-1, F.shape[2])
        y = M[ts].reshape(-1)
        return self._model().fit(X, y)

    def predict(self, t):
        t0 = (t // self.retrain) * self.retrain
        t0 = max(t0, self.min_train + 50)
        if t0 not in self.cache:
            self.cache[t0] = (self._train(self.FR, self.R, t0), self._train(self.FB, self.B, t0))
        mr, mb = self.cache[t0]
        pr = mr.predict_proba(self.FR[t])[:, 1]
        pb = mb.predict_proba(self.FB[t])[:, 1]
        return _norm(pr, 6), pb / pb.sum()


def all_models():
    return [Uniform(), Frequency(None), Frequency(30), Frequency(100), Cold(),
            ML("logreg"), ML("gbm")]
