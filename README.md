# AI Trading Bot — trading-xcamp-2026

An AI-powered stock trading system that predicts continuous 5-day forward returns for S&P 100 equities, ranks the universe weekly to run a market-neutral long/short strategy, backtests it with vectorbt, and executes paper trades via Alpaca.

## Architecture

```
download → features → train → backtest → paper-trade → dashboard
```

| Module | Path | Responsibility |
|--------|------|----------------|
| Data | `src/data/` | yfinance download, cleaning, parquet cache |
| Features | `src/features/` | pandas-ta indicators, cross-sectional labels, feature-family maps, Exp 3 multi-target panel |
| Models | `src/models/` | Pooled cross-sectional XGBoost / LightGBM / RF / CatBoost training & evaluation |
| Strategy | `src/strategy/` | Signal generation, risk limits, portfolio weights |
| Backtesting | `src/backtesting/` | vectorbt engine, Sharpe / drawdown metrics |
| Execution | `src/execution/` | Alpaca paper trading with dry-run mode |
| Dashboard | `src/dashboard/` | Streamlit + Plotly monitoring |

## Quick Start

### 1. Environment

Requires **Python 3.13+**.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configuration

Historical market data is downloaded from Yahoo Finance and does **not** require API keys.

For paper trading and live dashboard positions, copy and edit `.env`:

```bash
cp .env.example .env
# Edit .env with your Alpaca paper trading keys
```

Get free paper trading keys at [Alpaca Paper Dashboard](https://app.alpaca.markets/paper/dashboard/overview).

### 3. Run Pipeline

For a quick smoke test with 3 symbols, use the dev config:

```bash
python main.py --config configs/dev.yaml download
python main.py --config configs/dev.yaml features
python main.py --config configs/dev.yaml train
python main.py --config configs/dev.yaml backtest
python main.py --config configs/dev.yaml paper-trade --dry-run
```

Full S&P 100 pipeline:

```bash
python main.py pipeline

# Or step by step
python main.py download
python main.py features
python main.py train --model xgboost
python main.py backtest
python main.py paper-trade --dry-run
python main.py dashboard
```

### 4. Research Experiments

```bash
# Exp 1: model family comparison (XGB / LGBM / RF / CatBoost) + identical long-short backtests
python scripts/run_experiment1.py
python scripts/plot_experiment1.py

# Exp 2: feature-family ablation + importance pruning (frozen XGBoost)
python scripts/run_experiment2.py
python scripts/plot_experiment2.py

# Exp 3: target / label engineering (frozen XGBoost + Exp 2 top-5 features)
python scripts/run_experiment3.py
```

Design and results: [`docs/research_presentation.md`](docs/research_presentation.md), [`results/experiment_1/experiment_1_report.md`](results/experiment_1/experiment_1_report.md), [`results/experiment_2/experiment_2_report.md`](results/experiment_2/experiment_2_report.md), [`results/experiment_3/experiment_report.md`](results/experiment_3/experiment_report.md).

**Headline results (completed):**

| Experiment | Question | Winner / recommendation |
|------------|----------|-------------------------|
| Exp 1 (H1) | Which model family? | XGBoost (least-bad OOS trader; all models lost money on full features) |
| Exp 2 (H2) | Which features? | `top5` by test Sharpe (fragile val→test); prefer returns+volatility for coherence |
| Exp 3 (H3) | Which prediction target? | Keep absolute 5-day; 10d ranks better but does not trade better; 3d and CS-relative worse |

### 5. Tests

```bash
pytest tests/ -v
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `download` | Fetch daily OHLCV for S&P 100 from yfinance |
| `features` | Engineer indicators and cross-sectional 5-day labels |
| `train` | Train one pooled regressor (`xgboost`, `lightgbm`, `random_forest`, `catboost`) on 5-day forward returns |
| `backtest` | Run vectorbt backtest on test period |
| `paper-trade` | Rebalance paper portfolio (use `--dry-run` for safety) |
| `dashboard` | Launch Streamlit monitoring UI |
| `pipeline` | Run download → features → train → backtest |

## Configuration

- Global settings: [`configs/default.yaml`](configs/default.yaml)
- Dev smoke test (3 symbols): [`configs/dev.yaml`](configs/dev.yaml)
- S&P 100 tickers: [`configs/sp100_tickers.yaml`](configs/sp100_tickers.yaml)

Key parameters:

- **Data:** 2010–present daily bars; train through 2022, validate 2023–2024, out-of-sample test from 2025
- **Target:** Continuous 5-day forward return (`forward_return_5d`); models rank names by predicted return. Binary top-20% labels are retained only as a hit-rate evaluation helper (`labels.positive_quantile`). Experiment 3 confirmed keeping this target over 3-day, 10-day, and cross-sectional-relative alternatives.
- **Model scope:** One pooled regressor trained on all tickers simultaneously (predicted returns are directly comparable for ranking)
- **Strategy:** Weekly cross-sectional rank rebalance — long the top 10 symbols by predicted return score, short the bottom 10 (pure rank), 50% gross long / 50% gross short (market-neutral)
- **Backtest:** $100k initial, 1 bps commission + 5 bps slippage

### Train / validation / test splits

`calendar_split()` (`src/data/splits.py`) produces three chronological slices on the pooled panel:

- **Train** — all rows on or before `train_end_date` (2022-12-31), with the last `horizon_days` trading dates purged so forward labels cannot leak into validation.
- **Val** — `val_start_date`–`val_end_date` (2023–2024), similarly purged at the tail.
- **Test** — everything from `test_start_date` (2025-01-01) onward. This is the true out-of-sample holdout that the weekly long/short backtest runs against (`predictions_test.parquet`); the model never sees this data during training.

## Evaluation Metrics

**Regression / ranking helpers:** RMSE, MAE, R²; ROC-AUC / PR-AUC of predicted return vs binary top-20% label

**Cross-sectional:** Information Coefficient (Spearman), top-decile hit rate, mean forward return by prediction decile

**Trading:** Total Return, Annualized Return, Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor, Turnover

## Known Limitations

- S&P 100 list is point-in-time approximate (survivorship bias)
- Daily bars only (no intraday)
- Paper trading only — no live money
- Strong ranking/classification metrics do not guarantee profitability
- The long/short backtest applies symmetric commission/slippage bps to both legs but does not model short borrow fees, hard-to-borrow constraints, or margin interest
- Completed research experiments (Exp 1–3) find weak absolute OOS skill and val→test instability; model family, feature subset, and target engineering each move metrics, but only feature selection produced a clear trading-side lift under the frozen weekly long/short pipeline — and that lift remains fragile. See the experiment reports.

## Project Structure

```
src/
  data/          # download, clean, cache
  features/      # indicators, labels, feature families, Exp 3 multi-target panel
  models/        # train, evaluate, persist
  strategy/      # signals, risk, portfolio
  backtesting/   # vectorbt + metrics
  execution/     # Alpaca paper trading
  dashboard/     # Streamlit app
  visualization/ # Experiment 1/2/3 comparison figures
  utils/         # config, logging, paths
scripts/         # run_experiment1/2/3.py, plot helpers
tests/
configs/
docs/            # research presentation
results/         # experiment outputs and reports (experiment_1/2/3)
main.py
```

## License

Educational project — XCamp 2026.
