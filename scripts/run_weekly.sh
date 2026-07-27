#!/usr/bin/env bash
# Weekly pipeline runner — download, features, train, backtest, paper trade (dry-run)
set -euo pipefail

echo "=== AI Trading Bot Weekly Pipeline ==="
python main.py download
python main.py features
python main.py train
python main.py backtest
python main.py paper-trade --dry-run
echo "=== Pipeline complete ==="
