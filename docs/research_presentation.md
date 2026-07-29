# Research Presentation — Pooled Cross-Sectional Long-Short Strategy

This document describes the research question, hypotheses, experiments, and results for the project's AI-powered cross-sectional equity trading system: a pooled tree model that ranks S&P 100 stocks each week and drives a top-10 / bottom-10 long-short portfolio. Scope is limited to experiments that have been implemented and run in this repository.

## Overall Research Question

> Can a pooled, cross-sectional tree model trained on standard technical indicators generate economically exploitable weekly rankings of S&P 100 equities out-of-sample (predicting 5-day forward returns), and does the choice of model family or feature subset meaningfully affect the resulting top-10 / bottom-10 long-short strategy's risk-adjusted (Sharpe) performance?

---

## Hypotheses

### H1 — Model Family, Ranking Quality, and Strategy Performance

| Field                  | Detail                                                                                                                                                                                                                                                                 |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Null hypothesis        | Under matched universe, features, labels, calendar splits, weekly rebalance, and top-10 / bottom-10 construction, pooled XGBoost, LightGBM, Random Forest, and CatBoost do not differ meaningfully on ranking quality (IC, top-decile hit rate) or on long-short trading metrics (Sharpe, annualized return, max drawdown). |
| Alternative hypothesis | At least one model family is materially better on ranking quality and/or realized strategy performance.                                                                                                                                                                |
| Independent variables  | `model.type` (`xgboost`, `lightgbm`, `random_forest`, `catboost`) with `model.task: regression` — features, continuous target, splits, and portfolio construction held constant.                                                                                      |
| Dependent variables    | Ranking: Spearman IC (overall / mean daily), top-decile hit rate, ROC-AUC, PR-AUC. Trading: annualized return, Sharpe ratio, max drawdown, win rate, profit factor, turnover.                                                                                         |
| Evaluation metrics     | `evaluate_cross_sectional()` / `evaluate_regressor()` helpers; identical `run_backtest_on_predictions(..., strategy=long_short)`. Practical materiality: ΔSharpe ≥ 0.10 or Δ annualized return ≥ 2pp; ranking: Δ mean-daily IC ≥ 0.005; IC IR as a stability check. |
| Winner rule            | Primary: **test Sharpe**; ties → test mean-daily IC → top-decile hit rate.                                                                                                                                                                                             |
| Codebase support       | `scripts/run_experiment1.py`; `src/models/trainer.py::_build_regressor`; `src/models/cross_sectional.py`; `src/backtesting/engine.py::run_backtest_on_predictions`.                                                                                                     |

### H2 — Feature Family Ablation and Selection

| Field                  | Detail                                                                                                                                                                                                                                                                 |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Null hypothesis        | Under fixed pooled XGBoost, matched splits/labels, and identical weekly top-10 / bottom-10 long-short construction, ranking quality and trading performance do not differ meaningfully across feature subsets derived from the current technical set. |
| Alternative hypothesis | At least one reduced or regrouped feature set materially improves test Sharpe and/or mean-daily IC relative to the full 19-feature baseline.                                                                                                                           |
| Independent variables  | Feature arm / subset (Stage A family ablation; Stage B importance pruning). Model type, task, splits, labels, and portfolio construction held constant.                                                                                                                |
| Dependent variables    | Ranking: Spearman IC (overall / mean daily), IC IR, top-decile hit rate, ROC-AUC, PR-AUC. Trading: annualized return, Sharpe, max drawdown, win rate, profit factor, turnover.                                                                                         |
| Evaluation metrics     | Same Exp 1 bundle. Materiality: ΔSharpe ≥ 0.10 or Δ ann. return ≥ 2pp; Δ mean-daily IC ≥ 0.005.                                                                                                                                                                        |
| Winner rule            | Primary: **test Sharpe**; ties → test mean-daily IC → top-decile hit rate. Best Stage B arm is also reported by **val Sharpe**.                                                                                                                                       |
| Codebase support       | `scripts/run_experiment2.py`; `src/features/families.py`; `src/models/trainer.py` (`feature_columns` override); `src/backtesting/engine.py::run_backtest_on_predictions`.                                                                                               |

---

## Experiments

### Experiment 1 — Model Family, Ranking Quality, and Strategy Performance

**Status:** Complete. Results in [`results/experiment_1/`](../results/experiment_1/).

- **Objective:** Isolate whether the choice of pooled tree model family materially changes weekly cross-sectional ranking quality and the realized performance of an identical S&P 100 top-10 / bottom-10 long-short strategy.
- **Hypothesis tested:** H1.
- **Experimental setup:** Using `configs/default.yaml` (S&P 100, calendar splits 2010–2022 / 2023–2024 / 2025+, continuous `forward_return_5d`, standard technical features, `strategy.mode: long_short`, weekly rebalance, 10 long / 10 short), train four pooled regressors via `python scripts/run_experiment1.py`. Each model's weekly ranks feed the **same** trading pipeline (`run_backtest_on_predictions`). No threshold tuning or portfolio-construction changes.
- **Models compared:** Pooled XGBoost, LightGBM, Random Forest, CatBoost — matched depth/estimator budgets where applicable; ranking score = predicted 5-day forward return.
- **Dataset:** S&P 100 (`configs/sp100_tickers.yaml`), daily bars from 2010–present, splits per `configs/default.yaml`; last 5 train/val dates purged for label leakage.
- **Features used:** The standard 19 technical indicators from `get_feature_columns()` — 1/5/20-day returns, price/SMA and price/EMA ratios, MACD (line/signal/histogram, price-normalized), RSI(14), stochastic %K/%D, Bollinger Band width, ATR%, 20-day realized volatility, OBV z-score, and volume/SMA ratio.
- **Evaluation methodology:**
  1. Ranking quality on train/val/test: IC (overall), mean daily IC, top-decile hit rate, ROC-AUC, PR-AUC (score vs top-20% label).
  2. Trading performance on val and test via the shared long-short pipeline: annualized return, Sharpe, max drawdown, win rate, profit factor, turnover.
  3. Head-to-head comparison: best rankings? best portfolio? differences material? which model advances?
- **Success criteria:** A clear, reproducible model recommendation based primarily on **test Sharpe**.
- **Figures/tables produced:**
  - Tables: `metrics_by_model_split.csv`, `cross_sectional_by_model.csv`, `trading_by_model.csv`, `returns_by_model.csv`
  - Bars: test ranking/fit metrics; metrics by split; cross-sectional IC / hit rate; trading (Sharpe / ann. return / max DD); top-decile returns
  - Decile return curves; feature importance (all models + winner)
  - Single-metric panels label whether higher or lower is better
- **Key result:** All four models lose money on the 2025+ test long-short backtest. Winner by test Sharpe: **XGBoost** (Sharpe ≈ −0.11) — least-bad, not profitable. Random Forest leads ranking metrics (mean daily IC ≈ 0.011) but has worse test trading (Sharpe ≈ −0.29). Val→test performance collapses for every model. See [`results/experiment_1/experiment_1_report.md`](../results/experiment_1/experiment_1_report.md).

### Experiment 2 — Feature Family Ablation and Selection

**Status:** Complete. Results in [`results/experiment_2/`](../results/experiment_2/).

- **Objective:** Isolate whether the choice of feature subset (by economic family and/or importance pruning) materially changes weekly ranking quality and long-short portfolio performance, holding the model and trading pipeline fixed.
- **Hypothesis tested:** H2.
- **Experimental setup:** Fix pooled XGBoost and `configs/default.yaml` splits/labels/`long_short` weekly top-10 / bottom-10 construction. Run `python scripts/run_experiment2.py`.
  - **Stage A (family ablation):** `full`, `returns`, `trend`, `momentum`, `volatility`, `volume`, `returns_volatility`
  - **Stage B (importance prune from Full):** `top5`, `top10`, `cum80`
- **Models compared:** Feature arms only — same pooled XGBoost regressor; ranking score = predicted 5-day forward return.
- **Dataset:** Same S&P 100 panel and calendar splits as Experiment 1.
- **Features used:** Subsets of the standard 19 technicals (see `src/features/families.py`). Stage B subsets are importance-selected from the Full arm only (never from test).
- **Evaluation methodology:** Ranking + identical long-short backtest on val/test per arm; head-to-head vs Full.
- **Success criteria:** A clear feature-set recommendation based primarily on **test Sharpe**.
- **Figures/tables produced:**
  - Tables: `metrics_by_arm_split.csv`, `cross_sectional_by_arm.csv`, `trading_by_arm.csv`, `returns_by_arm.csv`
  - Bars for **val and test**: ranking/fit metrics; cross-sectional IC / hit rate; trading
  - Full-model feature importance (Stage B rationale)
  - Single-metric panels label whether higher or lower is better
- **Key result:** Feature composition matters more than model family. Full (19 features) remains negative OOS. Declared winner by test Sharpe: **top5** (Sharpe ≈ 0.56), but its val Sharpe was near zero (~0.09), so the win is fragile. Best Stage B by val Sharpe was **cum80**. Most coherent Stage A arm: **returns_volatility** (best val Sharpe; roughly flat test). Momentum/volume/trend-alone look harmful. See [`results/experiment_2/experiment_2_report.md`](../results/experiment_2/experiment_2_report.md).

---

## Presentation Ordering

1. **Experiment 1 — "Which model should drive the ranking strategy?"** (H1). Model family alone does not produce a profitable OOS long-short book; XGBoost advances as the least-bad trading model under the frozen pipeline.
2. **Experiment 2 — "Which features should that model use?"** (H2). With XGBoost frozen, feature subset choice moves portfolio outcomes more than model family did; smaller returns/volatility-oriented sets beat the full kitchen sink, but OOS winners remain fragile under a short 2025+ test window.

This progression — **model → features** — is what the codebase has completed and what the presentation should cover.
