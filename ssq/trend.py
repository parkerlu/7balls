"""走势技术分析检验: 把开奖指标当作"股价"序列, 用股票技术指标预测下一期。

序列: 和值 / 跨度 / 奇数个数 / 大号个数(>=17) / 蓝球
方法: 均值(理论) / 上期值 / 均线 MA5/10/20 / 动量(顺势) / 均值回归 / AR 自回归 / MACD 金叉死叉
评价: 预测误差 MAE; 方向 (下期比本期涨还是跌) 准确率; 与"永远猜理论均值"对比。
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats

from .models import load

START = 300


def series(df):
    r = df[[f"r{i}" for i in range(1, 7)]].to_numpy()
    return {
        "和值": r.sum(1).astype(float),
        "跨度": (r[:, -1] - r[:, 0]).astype(float),
        "奇数个数": (r % 2).sum(1).astype(float),
        "大号个数": (r >= 17).sum(1).astype(float),
        "蓝球": df["blue"].to_numpy().astype(float),
    }


def theory_means():
    c = np.array(list(combinations(range(1, 34), 6)))
    return {"和值": c.sum(1).mean(), "跨度": (c[:, -1] - c[:, 0]).mean(),
            "奇数个数": (c % 2).sum(1).mean(), "大号个数": (c >= 17).sum(1).mean(), "蓝球": 8.5}


def ema(x, n):
    a, out, e = 2 / (n + 1), np.empty_like(x), x[0]
    for i, v in enumerate(x):
        e = a * v + (1 - a) * e
        out[i] = e
    return out


def predictors(x, mu):
    """pred[name][t] = 用 x[:t] 对 x[t] 的预测"""
    N = len(x)
    prev = np.r_[np.nan, x[:-1]]
    ma = {n: np.r_[np.full(n, np.nan), np.convolve(x, np.ones(n) / n, "valid")[:-1]] for n in (5, 10, 20)}
    P = {"理论均值": np.full(N, mu), "上期值": prev}
    for n, m in ma.items():
        P[f"均线 MA{n}"] = m
    mom = prev + (prev - np.r_[np.nan, np.nan, x[:-2]])          # 顺势: 延续上期涨跌
    P["动量 (顺势)"] = mom
    P["均值回归"] = mu - 0.5 * (prev - mu)                       # 偏高就往回猜
    # AR(3) 滚动拟合
    ar = np.full(N, np.nan)
    for t in range(START, N):
        if (t - START) % 50 == 0:
            X = np.column_stack([x[2:t - 1], x[1:t - 2], x[0:t - 3], np.ones(t - 3)])
            coef = np.linalg.lstsq(X, x[3:t], rcond=None)[0]
        ar[t] = coef @ np.array([x[t - 1], x[t - 2], x[t - 3], 1.0])
    P["AR(3) 自回归"] = ar
    # MACD: DIF = EMA12 - EMA26, DEA = EMA9(DIF); 金叉看涨 -> 预测 上期值+1σ 方向
    dif = ema(x, 12) - ema(x, 26)
    dea = ema(dif, 9)
    sig = np.sign(dif - dea)
    P["MACD 信号"] = np.r_[np.nan, mu + sig[:-1] * x.std() * 0.3]
    return P


def run():
    df = load()
    S, M = series(df), theory_means()
    out = {}
    for name, x in S.items():
        mu = M[name]
        P = predictors(x, mu)
        ts = np.arange(START, len(x))
        y, prev = x[ts], x[ts - 1]
        up = y > prev
        flat = y == prev
        base = np.abs(y - mu)
        rows = []
        for pn, p in P.items():
            e = np.abs(y - p[ts])
            d = e - base
            pval = stats.ttest_1samp(d, 0, alternative="less").pvalue if pn != "理论均值" else 1.0
            guess_up = p[ts] > prev
            acc = float(((guess_up == up) & ~flat).sum() / (~flat).sum())
            rows.append({"name": pn, "mae": float(e.mean()), "vs_mean_pct": float(d.mean() / base.mean() * 100),
                         "p": float(pval), "dir_acc": acc})
        # 自相关
        z = x - x.mean()
        ac = [float(np.corrcoef(z[L:], z[:-L])[0, 1]) for L in (1, 2, 3)]
        out[name] = {"mu": float(mu), "mean_obs": float(x.mean()), "ac": ac, "rows": rows, "n": len(ts)}
    return out


if __name__ == "__main__":
    r = run()
    for name, d in r.items():
        print(f"\n【{name}】理论均值 {d['mu']:.2f} 实际均值 {d['mean_obs']:.2f}  滞后1/2/3期自相关 {np.round(d['ac'], 3).tolist()} (显著阈值 ±0.035)")
        print(f"  {'方法':<12}{'平均误差':>8}{'相比理论均值':>12}{'p':>8}{'涨跌方向准确率':>14}")
        for x in d["rows"]:
            print(f"  {x['name']:<12}{x['mae']:8.2f}{x['vs_mean_pct']:+11.1f}%{x['p']:8.3f}{x['dir_acc']:13.1%}")


# ---------------- 走势模型: 按均线目标挑选号码 ----------------
_COMBOS = None


def _combos():
    global _COMBOS
    if _COMBOS is None:
        c = np.array(list(combinations(range(1, 34), 6)), dtype=np.int8)
        _COMBOS = {"c": c, "和值": c.sum(1, dtype=np.int16), "跨度": (c[:, -1] - c[:, 0]).astype(np.int16),
                   "奇数个数": (c % 2).sum(1, dtype=np.int16), "大号个数": (c >= 17).sum(1, dtype=np.int16)}
    return _COMBOS


SCALE = {"和值": 19.5, "跨度": 4.9, "奇数个数": 1.1, "大号个数": 1.1}   # 各指标标准差, 用于归一化


def trend_predict(reds: np.ndarray, blues: np.ndarray, window: int = 10):
    """reds: (t,6) 历史红球, blues: (t,) 历史蓝球 -> (p_red[33], p_blue[16], targets)"""
    C = _combos()
    rec = reds[-window:]
    tgt = {"和值": rec.sum(1).mean(), "跨度": (rec[:, -1] - rec[:, 0]).mean(),
           "奇数个数": (rec % 2).sum(1).mean(), "大号个数": (rec >= 17).sum(1).mean()}
    dist = sum(((C[k] - tgt[k]) / SCALE[k]) ** 2 for k in tgt)
    # 近 window 期热度作为顺势加分 (很小, 只用于打破平局)
    hot = np.bincount(rec.ravel(), minlength=34)[1:] / window
    close = dist <= np.quantile(dist, 0.01)           # 最符合走势的 1% 组合
    freq = np.bincount(C["c"][close].ravel(), minlength=34)[1:].astype(float)
    p_red = freq / freq.sum() * 6
    score = dist - 0.05 * hot[C["c"] - 1].sum(1)
    best = C["c"][np.argmin(score)]
    # 让最优组合的 6 个号排在概率最前
    p_red = p_red * 0.999
    p_red[best - 1] = np.maximum(p_red[best - 1], p_red.max() * 1.0001)
    p_red = p_red / p_red.sum() * 6
    bt = blues[-window:].mean()
    d = np.abs(np.arange(1, 17) - bt)
    p_blue = np.exp(-d / 4)
    p_blue /= p_blue.sum()
    tgt["蓝球"] = bt
    return p_red, p_blue, {k: float(v) for k, v in tgt.items()}


def trend_backtest(start: int = START):
    df = load()
    R = df[[f"r{i}" for i in range(1, 7)]].to_numpy()
    Bl = df["blue"].to_numpy()
    pos, neg, bh, bn = [], [], [], []
    for t in range(start, len(df)):
        pr, pb, _ = trend_predict(R[:t], Bl[:t])
        o = np.argsort(-pr)
        y = set(R[t])
        pos.append(len(y & set(o[:6] + 1)))
        neg.append(len(y & set(o[-10:] + 1)))
        ob = np.argsort(-pb)
        bh.append(int(ob[0] + 1 == Bl[t]))
        bn.append(int(Bl[t] in set(ob[-5:] + 1)))
    pos, neg, bh, bn = map(np.array, (pos, neg, bh, bn))
    return {"n": len(pos),
            "pos": float(pos.mean()), "pos_p": float(stats.ttest_1samp(pos, 36 / 33, alternative="greater").pvalue),
            "neg": float(neg.mean()), "neg_p": float(stats.ttest_1samp(neg, 60 / 33, alternative="less").pvalue),
            "blue": float(bh.mean()), "blue_p": float(stats.binomtest(int(bh.sum()), len(bh), 1 / 16, alternative="greater").pvalue),
            "blue_neg": float(bn.mean())}
