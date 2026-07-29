# Research Presentation — Pooled Cross-Sectional Long-Short Strategy

This document describes the research question, hypotheses, experiments, and results for the project's AI-powered cross-sectional equity trading system: a pooled tree model that ranks S&P 100 stocks each week and drives a top-10 / bottom-10 long-short portfolio. Scope is limited to experiments that have been implemented and run in this repository.

## Overall Research Question

> Can a pooled, cross-sectional tree model trained on standard technical indicators generate economically exploitable weekly rankings of S&P 100 equities out-of-sample (predicting 5-day forward returns), and does the choice of model family, feature subset, or prediction target meaningfully affect the resulting top-10 / bottom-10 long-short strategy's risk-adjusted (Sharpe) performance?

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

### H3 — Target (Label) Engineering

| Field                  | Detail                                                                                                                                                                                                                                                                 |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Null hypothesis        | Under fixed pooled XGBoost, frozen Exp 2 top-5 features, matched universe/splits, and identical weekly top-10 / bottom-10 long-short construction, ranking quality and trading performance do not differ meaningfully across alternative prediction-target definitions versus the absolute 5-day forward return baseline. |
| Alternative hypothesis | At least one alternative target (3-day absolute, 10-day absolute, or cross-sectional relative 5-day) materially improves out-of-sample ranking quality **and** trading performance relative to the 5-day absolute baseline. |
| Independent variables  | Prediction target only: absolute `forward_return_5d` (baseline), absolute `forward_return_3d`, absolute `forward_return_10d`, cross-sectional relative `forward_return_5d_rel` (5d return − within-date median). Model, features, splits, and portfolio construction held constant. |
| Dependent variables    | Ranking (always vs common absolute 5d yardstick): Spearman IC (overall / mean daily), IC IR, top-decile hit rate, top−bottom decile spread, ROC-AUC, PR-AUC. Trading: annualized return, Sharpe, max drawdown, win rate, profit factor, turnover, total return. Own-target RMSE/MAE/R² are fit diagnostics only. |
| Evaluation metrics     | Same Exp 1/2 bundle. Materiality: ΔSharpe ≥ 0.10 or Δ ann. return ≥ 2pp; Δ mean-daily IC ≥ 0.005. Ranking metrics for all arms scored against absolute `forward_return_5d` (not each arm's own training horizon). |
| Winner rule            | Primary: **test Sharpe**; ties → test mean-daily IC (common 5d) → top-decile hit rate (common 5d). Best-by-val-Sharpe recorded separately. |
| Codebase support       | `scripts/run_experiment3.py`; `src/features/experiment3_targets.py`; `src/features/labels.py::compute_relative_forward_returns`; `src/models/trainer.py` (`target_col_override` / `purge_horizon_override`); `src/visualization/experiment3_plots.py`. |

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

### Experiment 3 — Target (Label) Engineering

**Status:** Complete. Results in [`results/experiment_3/`](../results/experiment_3/).

- **Objective:** Isolate whether changing the prediction target (label engineering) materially improves weekly ranking quality and long-short portfolio performance, holding the model, features, and trading pipeline fixed.
- **Hypothesis tested:** H3.
- **Experimental setup:** Fix pooled XGBoost and the Exp 2 `top5` feature set (`atr_pct`, `return_5d`, `volatility_20d`, `price_sma10_ratio`, `stoch_d`). Same S&P 100 universe, calendar splits, weekly `long_short` 10/10 construction, and costs as Exp 1/2. Run `python scripts/run_experiment3.py`. The only independent variable is the regression target.
  - **A (baseline):** absolute 5-day forward return
  - **B:** absolute 3-day forward return
  - **C:** absolute 10-day forward return
  - **D:** cross-sectional relative 5-day return (5d return − within-date median 5d return)
- **Models compared:** Target arms only — same pooled XGBoost regressor and same 5 features; ranking score = predicted value of each arm's training target.
- **Dataset:** Multi-target panel `data/processed/features_experiment3.parquet` (does not overwrite Exp 1/2 `features.parquet`). Binary eval `label` fixed at the 5-day horizon for all arms.
- **Fairness rule:** Ranking metrics (IC, hit rate, ROC/PR-AUC, decile spread) are always scored against absolute `forward_return_5d`, not each arm's own training horizon. Own-target RMSE/MAE/R² are fit diagnostics only. Trading uses the identical weekly long-short backtest for every arm.
- **Evaluation methodology:** Ranking + identical long-short backtest on val/test per target; ΔSharpe / ΔAnnRet / ΔIC vs baseline A; comparison to Exp 1 model-family deltas.
- **Success criteria:** Answer which target ranks best, which trades best OOS, whether improvements are material, whether target engineering beats Exp 1 model-family changes, and whether production should keep or switch the 5-day target.
- **Figures/tables produced:**
  - Tables: `metrics_by_target_split.csv`, `cross_sectional_by_target.csv`, `trading_by_target.csv`, `returns_by_target.csv`, `deltas_vs_baseline.csv`
  - Bars for **val and test**: trading (Sharpe / ann. return / max DD); cross-sectional IC / hit rate
  - Decile spread; feature importance by target; Δ-vs-baseline materiality chart
- **Key result:** H3 **not supported**. Winner by test Sharpe: **A (5d absolute)** (Sharpe ≈ 0.64, ann. return ≈ +7.2%). C (10d) has the best ranking quality (mean-daily IC ≈ 0.022, ΔIC ≈ +0.007 vs A) but slightly worse trading (Sharpe ≈ 0.56). B (3d) and D (5d relative) are materially worse on trading (ΔSharpe ≈ −0.57 and −0.95). Target engineering moved metrics more than Exp 1's model-family swap, but mostly in the wrong direction. **Keep the 5-day absolute production target.** See [`results/experiment_3/experiment_report.md`](../results/experiment_3/experiment_report.md).

---

## Presentation Ordering

1. **Experiment 1 — "Which model should drive the ranking strategy?"** (H1). Model family alone does not produce a profitable OOS long-short book; XGBoost advances as the least-bad trading model under the frozen pipeline.
2. **Experiment 2 — "Which features should that model use?"** (H2). With XGBoost frozen, feature subset choice moves portfolio outcomes more than model family did; smaller returns/volatility-oriented sets beat the full kitchen sink, but OOS winners remain fragile under a short 2025+ test window.
3. **Experiment 3 — "Which prediction target should that model fit?"** (H3). With XGBoost and top-5 features frozen, alternative targets (3d, 10d, CS-relative 5d) do not jointly improve ranking **and** trading vs the absolute 5-day baseline. The 10-day target ranks best but does not trade better under weekly rebalance; demeaning hurts. Keep the 5-day absolute target.

This progression — **model → features → target** — is what the codebase has completed and what the presentation should cover.
