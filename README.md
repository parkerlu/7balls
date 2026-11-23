# 双色球可预测性研究

用 2003 年以来的全部双色球开奖数据，检验两件事：

1. **号码能否预测**：多个模型做滚动回测，与均匀随机基线做配对显著性检验。2003–2018 年用于挑选模型，2019 年以后作为留出测试集。
2. **开奖是否回避热门号码**：用销量推算一、二等奖中奖注数的期望值，与实际注数对比；同时检验开奖号码是否回避生日号、连号等热门形态。

每期开奖前生成的预测存档在 `predictions/<期号>.json` 中，生成后不再覆盖。存档的提交时间和 SHA-256 校验值可以证明预测是在开奖前做出的。

网站：https://parkerlu.github.io/7balls/

## 运行

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m ssq.publish     # 抓数据、回测、检验、预测，写入 docs/data.json
./update.sh                         # 同上，并提交推送到 GitHub
```

## 结构

- `ssq/fetch.py`：抓取历史开奖数据
- `ssq/models.py`：各个模型（热号频率、冷号遗漏、逻辑回归、梯度提升树）
- `ssq/backtest.py`：滚动回测与显著性检验
- `ssq/audit.py`：开奖公正性检验
- `ssq/publish.py`：完整流程、预测存档与前瞻核对
- `docs/`：GitHub Pages 静态网站

仅用于学术研讨，不构成购彩建议。
