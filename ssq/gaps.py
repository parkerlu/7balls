"""间隙研究: 红球排序后相邻号码的间隙是否可预测。

间隙定义 (7 个, 和恒为 34):
  g1 = r1 (从 0 到第 1 个号), g2..g6 = r_i - r_{i-1}, g7 = 34 - r6 (尾部)
例: 2 8 11 12 23 32 -> 2 6 3 1 11 9 2
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

from .models import load

POS = ["首号", "1→2", "2→3", "3→4", "4→5", "5→6", "尾部"]
MAXG = 28  # 间隙最大可能值 (33-6+1)


def gaps_of(reds: np.ndarray) -> np.ndarray:
    z = np.zeros((len(reds), 1), int)
    full = np.hstack([z, reds, z + 34])
    return np.diff(full, axis=1)          # (N, 7)


def theory() -> np.ndarray:
    """枚举全部 C(33,6) 组合, 得到每个位置间隙的理论分布 P[pos, g]。"""
    combos = np.array(list(combinations(range(1, 34), 6)), dtype=np.int16)
    G = gaps_of(combos)
    P = np.zeros((7, MAXG + 1))
    for k in range(7):
        P[k] = np.bincount(G[:, k], minlength=MAXG + 1)[: MAXG + 1]
    return P / P.sum(1, keepdims=True)


def chi_fit(obs: np.ndarray, p: np.ndarray, min_exp=5):
    """把期望过小的尾部合并后做卡方拟合优度检验。"""
    n = len(obs)
    o = np.bincount(obs, minlength=len(p))[: len(p)].astype(float)
    e = p * n
    oo, ee, acc_o, acc_e = [], [], 0.0, 0.0
    for i in range(len(p)):
        acc_o += o[i]; acc_e += e[i]
        if acc_e >= min_exp:
            oo.append(acc_o); ee.append(acc_e); acc_o = acc_e = 0.0
    if acc_e > 0:
        oo[-1] += acc_o; ee[-1] += acc_e
    oo, ee = np.array(oo), np.array(ee)
    r = stats.chisquare(oo, ee * oo.sum() / ee.sum())
    return float(r.statistic), float(r.pvalue), len(oo) - 1


def lag_corr(G: np.ndarray, max_lag=10):
    """同位置自相关 + 上期任意位置 -> 本期任意位置的交叉相关。"""
    N = len(G)
    auto = np.zeros((7, max_lag))
    for k in range(7):
        x = G[:, k] - G[:, k].mean()
        for L in range(1, max_lag + 1):
            auto[k, L - 1] = np.corrcoef(x[L:], x[:-L])[0, 1]
    cross = np.corrcoef(G[1:].T, G[:-1].T)[:7, 7:]   # cross[i,j]: 本期 i 与上期 j
    return auto, cross, 1.96 / np.sqrt(N)


def backtest(G: np.ndarray, P: np.ndarray, start=300, lags=3, retrain=100):
    """每个位置: 用前 lags 期全部 7 个间隙做特征, 多分类逻辑回归预测本期间隙分布。
    与理论分布、经验分布比较对数损失和"猜中确切间隙"的比例。"""
    N = len(G)
    X = np.hstack([np.roll(G, L, axis=0) for L in range(1, lags + 1)]).astype(float) / 10.0
    res = []
    for k in range(7):
        ll_t, ll_e, ll_m, hit_t, hit_m = [], [], [], [], []
        model, classes = None, None
        for t in range(start, N):
            y = G[t, k]
            if model is None or (t - start) % retrain == 0:
                ys = G[lags:t, k]
                model = LogisticRegression(max_iter=2000, C=0.05).fit(X[lags:t], ys)
                classes = model.classes_
            pm = np.full(MAXG + 1, 1e-6)
            pm[classes] = model.predict_proba(X[t:t + 1])[0]
            pm /= pm.sum()
            emp = np.bincount(G[:t, k], minlength=MAXG + 1)[: MAXG + 1] + P[k] * 20
            emp = emp / emp.sum()
            ll_t.append(-np.log(max(P[k, y], 1e-9)))
            ll_e.append(-np.log(max(emp[y], 1e-9)))
            ll_m.append(-np.log(max(pm[y], 1e-9)))
            hit_t.append(int(np.argmax(P[k]) == y))
            hit_m.append(int(np.argmax(pm) == y))
        ll_t, ll_e, ll_m = map(np.array, (ll_t, ll_e, ll_m))
        d = ll_t - ll_m
        res.append({
            "pos": POS[k], "n": len(d),
            "ll_theory": float(ll_t.mean()), "ll_emp": float(ll_e.mean()), "ll_model": float(ll_m.mean()),
            "gain_pct": float(d.mean() / ll_t.mean() * 100),
            "p": float(stats.ttest_1samp(d, 0, alternative="greater").pvalue),
            "hit_theory": float(np.mean(hit_t)), "hit_model": float(np.mean(hit_m)),
        })
    return res


def run():
    df = load()
    reds = df[[f"r{i}" for i in range(1, 7)]].to_numpy()
    G = gaps_of(reds)
    P = theory()
    fit = []
    for k in range(7):
        chi, p, dof = chi_fit(G[:, k], P[k])
        fit.append({"pos": POS[k], "mean_obs": float(G[:, k].mean()),
                    "mean_exp": float((P[k] * np.arange(MAXG + 1)).sum()),
                    "mode": int(np.argmax(P[k])), "p_mode": float(P[k].max()),
                    "chi": chi, "dof": dof, "p": p})
    auto, cross, thr = lag_corr(G)
    bt = backtest(G, P)
    return {"N": len(G), "last": {"issue": df.issue.iloc[-1], "reds": reds[-1].tolist(), "gaps": G[-1].tolist()},
            "theory": P.tolist(), "fit": fit, "auto": auto.tolist(), "cross": cross.tolist(),
            "thr": float(thr), "backtest": bt}


if __name__ == "__main__":
    r = run()
    print("最新一期", r["last"])
    print("\n[1] 分布检验 (实际 vs 理论)")
    for f in r["fit"]:
        print(f"  {f['pos']:<4} 均值 {f['mean_obs']:.3f}/{f['mean_exp']:.3f}  最常见间隙 {f['mode']} (理论概率 {f['p_mode']:.1%})  χ²={f['chi']:.1f} df={f['dof']} p={f['p']:.3f}")
    auto, cross, thr = np.array(r["auto"]), np.array(r["cross"]), r["thr"]
    print(f"\n[2] 序列相关 (显著阈值 |r| > {thr:.3f})")
    print("  同位置 1-10 期自相关最大绝对值:", np.round(np.abs(auto).max(1), 3).tolist())
    n_sig = int((np.abs(auto) > thr).sum())
    print(f"  70 个自相关中超过阈值的: {n_sig} 个 (纯随机预期约 3.5 个)")
    print(f"  49 个交叉相关最大绝对值 {np.abs(cross).max():.3f}, 超过阈值 {int((np.abs(cross) > thr).sum())} 个 (随机预期约 2.5 个)")
    print("\n[3] 滚动回测 (逻辑回归, 特征=前3期全部间隙)")
    for b in r["backtest"]:
        print(f"  {b['pos']:<4} 对数损失 理论 {b['ll_theory']:.4f} 经验 {b['ll_emp']:.4f} 模型 {b['ll_model']:.4f}  增益 {b['gain_pct']:+.3f}% p={b['p']:.3f}  猜中确切间隙 理论众数 {b['hit_theory']:.1%} 模型 {b['hit_model']:.1%}")
