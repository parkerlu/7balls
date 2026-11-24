"""开奖公正性检验: 检验"开奖号码被挑选成少有人中"这一假设留下的可观测痕迹。"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .models import NB, NR, load, matrices

COMBOS = 17_721_088           # C(33,6) * 16
P_FIRST = 1 / COMBOS
P_SECOND = 15 / COMBOS        # 6 红 + 蓝不中


def _features(reds: np.ndarray, blue: np.ndarray) -> dict[str, np.ndarray]:
    """reds: (N,6) 升序, blue: (N,)"""
    consec = (np.diff(reds, axis=1) == 1).any(axis=1)
    return {
        "全部红球 ≤ 31 (生日号)": reds[:, -1] <= 31,
        "含连号": consec,
        "红球中 ≥3 个 ≤12 (月份号)": (reds <= 12).sum(1) >= 3,
        "蓝球 ≤ 12": blue <= 12,
        "含 6 或 8 (吉利号)": ((reds == 6) | (reds == 8)).any(1),
    }


def run(seed: int = 7) -> dict:
    df = load()
    df = df[df["sales"] > 0].reset_index(drop=True)   # 刚开奖、销量和中奖注数尚未公布的期次不参与检验
    R, B = matrices(df)
    N = len(df)

    # 1. 号码均匀性 (卡方检验)
    rc, bc = R.sum(0), B.sum(0)
    chi_r = stats.chisquare(rc)
    chi_b = stats.chisquare(bc)

    # 2. 一等奖 / 二等奖注数 vs 销量推算的期望值
    tickets = df["sales"].to_numpy() / 2
    lam1, lam2 = tickets * P_FIRST, tickets * P_SECOND
    f1, f2 = df["first_n"].to_numpy(), df["second_n"].to_numpy()
    zero_obs = int((f1 == 0).sum())
    zero_exp = float(np.exp(-lam1).sum())
    by_year = []
    years = df["date"].str[:4].to_numpy()
    for y in sorted(set(years)):
        m = years == y
        by_year.append({"year": y, "draws": int(m.sum()),
                        "first_obs": int(f1[m].sum()), "first_exp": float(lam1[m].sum()),
                        "second_obs": int(f2[m].sum()), "second_exp": float(lam2[m].sum()),
                        "zero_obs": int((f1[m] == 0).sum()), "zero_exp": float(np.exp(-lam1[m]).sum())})

    # 3. 开奖号码是否回避热门号码形态 (与 20 万次随机模拟比较)
    rng = np.random.default_rng(seed)
    S = 200_000
    sim_r = np.sort(np.argsort(rng.random((S, NR)), axis=1)[:, :6] + 1, axis=1)
    sim_b = rng.integers(1, NB + 1, S)
    obs = _features(df[[f"r{i}" for i in range(1, 7)]].to_numpy(), df["blue"].to_numpy())
    sim = _features(sim_r, sim_b)
    patterns = []
    for k in obs:
        p0 = float(sim[k].mean())
        k_obs = int(obs[k].sum())
        p = stats.binomtest(k_obs, N, p0, alternative="less").pvalue
        patterns.append({"name": k, "obs": k_obs / N, "exp": p0, "p_less": float(p)})

    # 一等奖 总观测/总期望 的 bootstrap 95% 区间 (按期重抽样, 自带过度离散)
    idx = rng.integers(0, N, (5000, N))
    boot = f1[idx].sum(1) / lam1[idx].sum(1)
    ratio_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    disp = float(f1.var() / f1.mean())

    return {
        "n": N,
        "first_ratio": float(f1.sum() / lam1.sum()), "first_ratio_ci": ratio_ci,
        "first_dispersion": disp,
        "first_median_ratio": float(np.median(f1 / lam1)),
        "red_counts": rc.tolist(), "blue_counts": bc.tolist(),
        "chi_red": {"stat": float(chi_r.statistic), "p": float(chi_r.pvalue)},
        "chi_blue": {"stat": float(chi_b.statistic), "p": float(chi_b.pvalue)},
        "first": {"obs": int(f1.sum()), "exp": float(lam1.sum()),
                  "zero_obs": zero_obs, "zero_exp": zero_exp},
        "second": {"obs": int(f2.sum()), "exp": float(lam2.sum())},
        "by_year": by_year,
        "patterns": patterns,
    }


if __name__ == "__main__":
    import json
    r = run()
    r.pop("by_year"); r.pop("red_counts"); r.pop("blue_counts")
    print(json.dumps(r, ensure_ascii=False, indent=1))
