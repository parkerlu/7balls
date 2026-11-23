"""滚动回测 (walk-forward): 每期只用之前的数据预测, 与均匀随机基线做配对显著性检验。

验证段 (2003-2018) 用于挑选正式模型; 测试段 (2019-至今) 只看一次, 作为留出检验。
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .models import NB, NR, all_models, load, matrices

START = 300          # 前 300 期作为冷启动, 不计分
TEST_FROM = "2019001"
K_POS, K_NEG, KB_NEG = 6, 10, 5


def score(model, R, B, ts):
    out = {k: [] for k in ["red_ll", "blue_ll", "top6", "bot10", "b_top1", "b_top3", "b_bot5"]}
    for t in ts:
        pr, pb = model.predict(t)
        y, b = R[t], int(np.argmax(B[t]))
        pr_c = np.clip(pr, 1e-9, 1 - 1e-9)
        out["red_ll"].append(-np.mean(y * np.log(pr_c) + (1 - y) * np.log(1 - pr_c)))
        out["blue_ll"].append(-np.log(max(pb[b], 1e-12)))
        order = np.argsort(-pr)
        out["top6"].append(y[order[:K_POS]].sum())
        out["bot10"].append(y[order[-K_NEG:]].sum())
        border = np.argsort(-pb)
        out["b_top1"].append(int(b == border[0]))
        out["b_top3"].append(int(b in border[:3]))
        out["b_bot5"].append(int(b in border[-KB_NEG:]))
    return {k: np.asarray(v, float) for k, v in out.items()}


BASE = {"top6": K_POS * 6 / NR, "bot10": K_NEG * 6 / NR,
        "b_top1": 1 / NB, "b_top3": 3 / NB, "b_bot5": KB_NEG / NB}
# 越大越好的指标; bot10 / b_bot5 是"反向排除", 越小越好
HIGHER_BETTER = {"top6": True, "b_top1": True, "b_top3": True, "bot10": False, "b_bot5": False}


def summarize(s, base_s):
    res = {"n": int(len(s["red_ll"]))}
    for k in ("red_ll", "blue_ll"):
        d = base_s[k] - s[k]                     # >0 表示比随机好
        p = stats.ttest_1samp(d, 0, alternative="greater").pvalue if d.std() > 0 else 1.0
        res[k] = {"value": float(s[k].mean()), "base": float(base_s[k].mean()),
                  "gain_pct": float(d.mean() / base_s[k].mean() * 100), "p": float(p)}
    for k, b in BASE.items():
        alt = "greater" if HIGHER_BETTER[k] else "less"
        p = stats.ttest_1samp(s[k], b, alternative=alt).pvalue
        res[k] = {"value": float(s[k].mean()), "base": b, "p": float(p)}
    return res


def run():
    df = load()
    R, B = matrices(df)
    T = len(df)
    split = int(np.searchsorted(df["issue"].to_numpy(), TEST_FROM))
    val_ts, test_ts = range(START, split), range(split, T)
    models = [m.fit(R, B) for m in all_models()]
    scores = {m.name: (score(m, R, B, val_ts), score(m, R, B, test_ts)) for m in models}
    base_v, base_t = scores["uniform"]
    results = []
    for m in models:
        sv, st = scores[m.name]
        results.append({"name": m.name, "label": m.label,
                        "val": summarize(sv, base_v), "test": summarize(st, base_t)})
    cands = [r for r in results if r["name"] != "uniform"]
    best_red = min(cands, key=lambda r: r["val"]["red_ll"]["value"])["name"]
    best_blue = min(cands, key=lambda r: r["val"]["blue_ll"]["value"])["name"]
    meta = {"start_issue": df["issue"][START], "split_issue": df["issue"][split],
            "end_issue": df["issue"].iloc[-1], "n_models": len(cands),
            "best_red": best_red, "best_blue": best_blue}
    return results, meta, models, df


if __name__ == "__main__":
    import json
    res, meta, *_ = run()
    print(json.dumps(meta, ensure_ascii=False))
    for r in res:
        v, t = r["val"], r["test"]
        print(f"{r['label']:<22} 红LL增益 val {v['red_ll']['gain_pct']:+.3f}% p={v['red_ll']['p']:.3f} | "
              f"test {t['red_ll']['gain_pct']:+.3f}% p={t['red_ll']['p']:.3f} | "
              f"蓝LL test {t['blue_ll']['gain_pct']:+.3f}% p={t['blue_ll']['p']:.3f} | "
              f"top6 {t['top6']['value']:.3f}/{t['top6']['base']:.3f} | bot10 {t['bot10']['value']:.3f}/{t['bot10']['base']:.3f} | "
              f"蓝top1 {t['b_top1']['value']:.3f} | 蓝bot5 {t['b_bot5']['value']:.3f}/{t['b_bot5']['base']:.3f}")
