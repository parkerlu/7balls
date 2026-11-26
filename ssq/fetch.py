"""抓取双色球全部历史开奖数据 -> data/draws.csv

数据源: datachart.500.com 历史开奖表(含销量、奖池、一二等奖注数)。
"""
from __future__ import annotations

import csv
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "draws.csv"
URL = "https://datachart.500.com/ssq/history/newinc/history.php?start=03001&end=99999"

FIELDS = ["issue", "date", "r1", "r2", "r3", "r4", "r5", "r6", "blue",
          "pool", "first_n", "first_prize", "second_n", "second_prize", "sales"]


def _int(s: str) -> int:
    s = s.replace(",", "").strip()
    return int(s) if s and s.lstrip("-").isdigit() else 0


def fetch() -> list[dict]:
    """全量页面有缓存会滞后, 再按期号范围补抓最近几期并合并。"""
    rows = {d["issue"]: d for d in _fetch(URL)}
    last = max(rows)
    recent = URL.replace("start=03001", f"start={last[2:4]}{int(last[4:]) - 5:03d}" if int(last[4:]) > 5 else f"start={last[2:4]}001")
    for d in _fetch(recent):
        rows[d["issue"]] = d
    return sorted(rows.values(), key=lambda d: d["issue"])


def _fetch(url: str) -> list[dict]:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "lxml")
    rows = []
    for tr in soup.select("tr.t_tr1"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        # issue, r1..r6, blue, (快乐星期天/空), pool, first_n, first_prize, second_n, second_prize, sales, date
        if len(tds) < 16:
            continue
        issue = tds[0]
        reds = sorted(int(x) for x in tds[1:7])
        rows.append({
            "issue": "20" + issue if len(issue) == 5 else issue,
            "date": tds[15],
            **{f"r{i+1}": reds[i] for i in range(6)},
            "blue": int(tds[7]),
            "pool": _int(tds[9]),
            "first_n": _int(tds[10]),
            "first_prize": _int(tds[11]),
            "second_n": _int(tds[12]),
            "second_prize": _int(tds[13]),
            "sales": _int(tds[14]),
        })
    rows.sort(key=lambda d: d["issue"])
    return rows


def main() -> None:
    rows = fetch()
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} 期, {rows[0]['issue']} ~ {rows[-1]['issue']} ({rows[-1]['date']}) -> {OUT}")


if __name__ == "__main__":
    main()
