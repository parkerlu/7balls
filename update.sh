#!/bin/bash
# 抓取最新开奖 -> 回测 -> 生成并存档下一期预测 -> 推送到 GitHub (提交时间即预测时间证明)
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python -m ssq.publish
git add data predictions docs
if ! git diff --cached --quiet; then
  ISSUE=$(ls predictions/*.json | sort | tail -1 | xargs basename | sed "s/.json//")
  git commit -q -m "预测 ${ISSUE} 并更新数据"
  git push -q origin main
  echo "已推送: ${ISSUE}"
else
  echo "无变化"
fi
