"""完整流程: 抓数据 -> 回测 -> 检验 -> 生成下一期预测并存档 -> 核对历史预测 -> 写 docs/data.json

预测存档 predictions/<期号>.json 一经生成永不覆盖; 推送到 GitHub 后由提交时间证明"开奖前已生成"。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from . import audit, backtest, fetch, trend
from .models import ML, NB, NR, Frequency, Uniform, load, matrices

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "predictions"
CTRL = PRED / "control"   # 对照模型 (不含号码大小特征), 同样只存不改
CONTROL = "ml_logreg_nonum"
TREND_DIR = PRED / "trend"   # 走势模型 (均线目标), 同样只存不改
DOCS = ROOT / "docs"
OFFICIAL = "ml_logreg"   # 2003-2018 验证段选出的正式模型, 预注册后不再更改
DRAW_WEEKDAYS = {1, 3, 6}  # 周二 周四 周日


def next_draw(last_issue: str, last_date: str) -> tuple[str, str]:
    d = dt.date.fromisoformat(last_date) + dt.timedelta(days=1)
    while d.weekday() not in DRAW_WEEKDAYS:
        d += dt.timedelta(days=1)
    year = int(last_issue[:4])
    issue = f"{d.year}001" if d.year != year else str(int(last_issue) + 1)
    return issue, d.isoformat()


def picks(pr: np.ndarray, pb: np.ndarray) -> dict:
    ro, bo = np.argsort(-pr), np.argsort(-pb)
    return {
        "pos_red": sorted(int(i) + 1 for i in ro[:6]),
        "pos_blue": int(bo[0]) + 1,
        "pos_blue3": [int(i) + 1 for i in bo[:3]],
        "neg_red": sorted(int(i) + 1 for i in ro[-10:]),
        "neg_blue": sorted(int(i) + 1 for i in bo[-5:]),
    }


def make_prediction(df, R, B, model=None, name=OFFICIAL) -> dict:
    T = len(df)
    issue, date = next_draw(df["issue"].iloc[-1], df["date"].iloc[-1])
    official = (model or ML("logreg", retrain=1)).fit(R, B)
    pr, pb = official.predict(T)
    others = {}
    for m in [Frequency(None), Frequency(100), ML("gbm", retrain=1)]:
        m.fit(R, B)
        a, b = m.predict(T)
        others[m.name] = {"label": m.label, "red": a.round(5).tolist(), "blue": b.round(5).tolist()}
    body = {
        "issue": issue, "draw_date": date,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "data_through": df["issue"].iloc[-1],
        "model": name,
        "red": pr.round(5).tolist(), "blue": pb.round(5).tolist(),
        "picks": picks(pr, pb),
    }
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body, others


def make_trend(df) -> dict:
    reds = df[[f"r{i}" for i in range(1, 7)]].to_numpy()
    pr, pb, tgt = trend.trend_predict(reds, df["blue"].to_numpy())
    issue, date = next_draw(df["issue"].iloc[-1], df["date"].iloc[-1])
    body = {"issue": issue, "draw_date": date,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "data_through": df["issue"].iloc[-1], "model": "trend_ma10",
            "targets": {k: round(v, 2) for k, v in tgt.items()},
            "red": pr.round(5).tolist(), "blue": pb.round(5).tolist(), "picks": picks(pr, pb)}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def archive(pred: dict, folder: Path = PRED) -> bool:
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / f"{pred['issue']}.json"
    if f.exists():
        return False
    f.write_text(json.dumps(pred, ensure_ascii=False, indent=1))
    return True


def score_archive(df, folder: Path = PRED) -> list[dict]:
    by_issue = {r["issue"]: r for r in df.to_dict("records")}
    rows = []
    for f in sorted(folder.glob("*.json")):
        p = json.loads(f.read_text())
        d = by_issue.get(p["issue"])
        row = {"issue": p["issue"], "draw_date": p["draw_date"], "generated_at": p["generated_at"],
               "picks": p["picks"], "sha256": p["sha256"], "drawn": d is not None}
        if d:
            reds = [d[f"r{i}"] for i in range(1, 7)]
            pr, pb = np.array(p["red"]), np.array(p["blue"])
            y = np.zeros(NR); y[np.array(reds) - 1] = 1
            prc = np.clip(pr, 1e-9, 1 - 1e-9)
            ll = -np.mean(y * np.log(prc) + (1 - y) * np.log(1 - prc))
            u = 6 / NR
            ll_u = -np.mean(y * np.log(u) + (1 - y) * np.log(1 - u))
            row.update({
                "red": p["red"], "blue_p": p["blue"],
                "reds": reds, "blue": d["blue"],
                "pos_hits": len(set(reds) & set(p["picks"]["pos_red"])),
                "pos_blue_hit": d["blue"] == p["picks"]["pos_blue"],
                "neg_hits": len(set(reds) & set(p["picks"]["neg_red"])),
                "neg_blue_hit": d["blue"] in p["picks"]["neg_blue"],
                "red_ll_gain": float(ll_u - ll),
                "blue_ll_gain": float(np.log(NB) + np.log(pb[d["blue"] - 1])),
            })
        rows.append(row)
    return rows


def main(refetch: bool = True) -> None:
    if refetch:
        fetch.main()
    df = load(str(ROOT / "data" / "draws.csv"))
    R, B = matrices(df)
    results, meta, *_ = backtest.run()
    aud = audit.run()
    pred, others = make_prediction(df, R, B)
    new = archive(pred)
    current = json.loads((PRED / f"{pred['issue']}.json").read_text())
    cpred, _ = make_prediction(df, R, B, ML("logreg", retrain=1, use_num=False), CONTROL)
    archive(cpred, CTRL)
    control = json.loads((CTRL / f"{cpred['issue']}.json").read_text())
    tpred = make_trend(df)
    archive(tpred, TREND_DIR)
    trend_next = json.loads((TREND_DIR / f"{tpred['issue']}.json").read_text())
    recent = df.tail(20).iloc[::-1]
    data = {
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "meta": meta, "official": OFFICIAL,
        "next": current, "others": others, "control": CONTROL,
        "control_next": control, "control_prospective": score_archive(df, CTRL),
        "trend_next": trend_next, "trend_prospective": score_archive(df, TREND_DIR),
        "trend_backtest": trend.trend_backtest(), "trend_series": trend.run(),
        "backtest": results, "audit": aud,
        "prospective": score_archive(df),
        "recent": [{"issue": r["issue"], "date": r["date"],
                    "reds": [r[f"r{i}"] for i in range(1, 7)], "blue": r["blue"],
                    "sales": r["sales"], "first_n": r["first_n"]} for r in recent.to_dict("records")],
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data.json").write_text(json.dumps(data, ensure_ascii=False))
    print(f"下一期 {pred['issue']} ({pred['draw_date']}) 预测{'已新建存档' if new else '已存在, 未覆盖'}")
    print("正向:", current["picks"]["pos_red"], "+", current["picks"]["pos_blue"])
    print("反向:", current["picks"]["neg_red"], "蓝", current["picks"]["neg_blue"])


if __name__ == "__main__":
    import sys
    main(refetch="--no-fetch" not in sys.argv)
