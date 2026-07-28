# Research Presentation — Pooled Cross-Sectional Long-Short Strategy

This document lays out the research question, hypotheses, and experiments for a presentation on the project's AI-powered cross-sectional equity trading system: a pooled tree model that ranks S&P 100 stocks each week and drives a top-10 / bottom-10 long-short portfolio. Everything below is scoped to what the current codebase can actually run — no invented features, models, or data sources.

## Overall Research Question

> Can a pooled, cross-sectional tree model trained on standard technical indicators generate economically exploitable weekly rankings of S&P 100 equities out-of-sample (predicting 5-day forward returns), and does the choice of model family, feature subset, probability calibration, and confidence-based trade gating meaningfully affect the resulting top-10 / bottom-10 long-short strategy's risk-adjusted (Sharpe) performance?

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
| Expected outcome       | Modest absolute skill given noisy daily technical features; boosted trees may edge Random Forest on ranking, but trading metrics decide the production model.                                                                                                          |
| Codebase support       | `scripts/run_experiment1.py`; `src/models/trainer.py::_build_regressor`; `src/models/evaluator.py`; `src/models/cross_sectional.py`; `src/backtesting/engine.py::run_backtest_on_predictions`.                                                                         |

### H2 — Feature Family Ablation and Selection

| Field                  | Detail                                                                                                                                                                                                                                                                 |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Null hypothesis        | Under fixed pooled XGBoost, matched splits/labels, and identical weekly top-10 / bottom-10 long-short construction, ranking quality and trading performance do not differ meaningfully across feature subsets derived from the current technical set. |
| Alternative hypothesis | At least one reduced or regrouped feature set materially improves test Sharpe and/or mean-daily IC relative to the full 19-feature baseline.                                                                                                                           |
| Independent variables  | Feature arm / subset (Stage A family ablation; Stage B importance pruning). Model type, task, splits, labels, and portfolio construction held constant.                                                                                                                |
| Dependent variables    | Ranking: Spearman IC (overall / mean daily), IC IR, top-decile hit rate, ROC-AUC, PR-AUC. Trading: annualized return, Sharpe, max drawdown, win rate, profit factor, turnover.                                                                                         |
| Evaluation metrics     | Same Exp 1 bundle via `evaluate_cross_sectional` / `evaluate_regressor` and identical `run_backtest_on_predictions(..., strategy=long_short)`. Materiality: ΔSharpe ≥ 0.10 or Δ ann. return ≥ 2pp; Δ mean-daily IC ≥ 0.005.                                           |
| Expected outcome       | The full set is likely redundant; a smaller economically coherent subset (e.g. returns + volatility) may match or beat Full by reducing overfitting. Absolute skill may remain modest.                                                                                 |
| Codebase support       | `scripts/run_experiment2.py`; `src/features/families.py`; `src/models/trainer.py` (`feature_columns` override); `src/backtesting/engine.py::run_backtest_on_predictions`.                                                                                               |

### H3 — Calibration Quality

| Field                  | Detail                                                                                                                                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Null hypothesis        | Platt scaling and isotonic regression do not reduce Brier score or Expected Calibration Error (ECE) relative to raw model probability on the test split.                                                           |
| Alternative hypothesis | At least one calibrator materially reduces Brier and/or ECE (per the codebase's own materiality thresholds: ΔBrier > 0.005 or ΔECE > 0.01), while rank order (IC) is preserved because the transform is monotonic. |
| Independent variables  | Probability column used: `probability` (raw), `probability_platt`, `probability_isotonic`.                                                                                                                         |
| Dependent variables    | Brier score, ECE, reliability-bin gap, IC (as an invariance check).                                                                                                                                                |
| Evaluation metrics     | `evaluate_classifier()` (`brier_score`, `ece`, `calibration_bins`); `information_coefficient()`.                                                                                                                   |
| Expected outcome       | Tree ensembles tend to be overconfident at the probability extremes; isotonic/Platt calibration should lower Brier/ECE without materially changing IC.                                                             |
| Codebase support       | `src/models/calibration.py`; `src/models/calibration_analysis.py::run_calibration_analysis`; `tests/test_calibration.py`; `tests/test_evaluator.py` (ECE and calibration-bin tests).                               |

### H4 — Confidence-Gated Trading vs. Pure Ranking

| Field                  | Detail                                                                                                                                                                                                                                                                                                                           |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Null hypothesis        | The confidence-gated strategy (`long_short_confidence`, thresholds selected on validation and frozen on test) does not beat the pure-ranking strategy (`long_short`) on test Sharpe ratio, or only appears to beat it with fewer than 10 trades.                                                                                 |
| Alternative hypothesis | `long_short_confidence` beats `long_short` on test Sharpe ratio while producing at least 10 trades in the validation threshold-selection window.                                                                                                                                                                                 |
| Independent variables  | `strategy.mode` (`long_short` vs. `long_short_confidence`); threshold pair swept over `{(0.60, 0.40), (0.65, 0.35), (0.70, 0.30)}`.                                                                                                                                                                                              |
| Dependent variables    | Sharpe ratio, total trades, win rate, max drawdown, turnover (test split).                                                                                                                                                                                                                                                       |
| Evaluation metrics     | `compute_backtest_metrics()` via `run_backtest_on_predictions()`.                                                                                                                                                                                                                                                                |
| Expected outcome       | Genuinely uncertain a priori — the codebase already encodes this as an empirical decision (`_build_recommendation`). The presentation should report whichever direction the data supports, noting that the 10-trade floor exists specifically to guard against spurious Sharpe spikes from tight thresholds with too few trades. |
| Codebase support       | `src/models/calibration_analysis.py` (the full A/B design is already implemented); `main.py calibrate`; `tests/test_calibration_analysis.py`.                                                                                                                                                                                    |

---

## Experiments

Each experiment maps directly onto one hypothesis.

### Experiment 1 — Model Family, Ranking Quality, and Strategy Performance

- **Objective:** Isolate whether the choice of pooled tree model family materially changes weekly cross-sectional ranking quality and the realized performance of an identical S&P 100 top-10 / bottom-10 long-short strategy.
- **Hypothesis tested:** H1.
- **Experimental setup:** Using `configs/default.yaml` (S&P 100, calendar splits 2010–2022 / 2023–2024 / 2025+, continuous `forward_return_5d`, standard technical features, `strategy.mode: long_short`, weekly rebalance, 10 long / 10 short), train four pooled regressors via `python scripts/run_experiment1.py`. Each model's weekly ranks feed the **same** trading pipeline (`run_backtest_on_predictions`). No threshold tuning, calibration, or confidence gating.
- **Models compared:** Pooled XGBoost, LightGBM, Random Forest, CatBoost — matched depth/estimator budgets where applicable; ranking score = predicted 5-day forward return.
- **Dataset:** S&P 100 (`configs/sp100_tickers.yaml`), daily bars from 2010–present, splits per `configs/default.yaml`; last 5 train/val dates purged for label leakage.
- **Features used:** The standard technical indicators from `get_feature_columns()` — 1/5/20-day returns, price/SMA and price/EMA ratios, MACD (line/signal/histogram, price-normalized), RSI(14), stochastic %K/%D, Bollinger Band width, ATR%, 20-day realized volatility, OBV z-score, and volume/SMA ratio.
- **Evaluation methodology:**
  1. **Model / ranking quality** on train/val/test: IC (overall), mean daily IC, top-decile hit rate, ROC-AUC, PR-AUC (score vs top-20% label).
  2. **Trading performance** on val and test via the shared long-short pipeline: annualized return, Sharpe, max drawdown, win rate, profit factor, turnover.
  3. Head-to-head comparison answering: best rankings? best portfolio? differences material (ΔSharpe / ΔIC thresholds + IC IR)? which model advances?
- **Success criteria:** A clear, reproducible model recommendation for later experiments based primarily on **test Sharpe**, with ranking metrics reported as explanatory evidence (ties broken by test mean-daily IC, then top-decile hit rate). Absolute skill may be modest; the experiment succeeds if the comparison isolates model-family effects under a fixed strategy.
- **Expected figures/tables for the presentation:**
  - Table: ranking metrics (IC, mean daily IC, top-decile hit rate, ROC-AUC, PR-AUC) × model × split.
  - Table: trading metrics (ann. return, Sharpe, max DD, win rate, profit factor, turnover) × model × (val/test).
  - Bars: test Sharpe and annualized return by model.
  - Cross-sectional IC / hit-rate comparison.
  - Decile return curves (ranking diagnostics).
  - Feature importance for the winning model.
  - Optional equity/drawdown overlay for all four models on test.

### Experiment 2 — Feature Family Ablation and Selection

- **Objective:** Isolate whether the choice of feature subset (by economic family and/or importance pruning) materially changes weekly ranking quality and the realized performance of an identical S&P 100 top-10 / bottom-10 long-short strategy, holding the model and trading pipeline fixed.
- **Hypothesis tested:** H2.
- **Experimental setup:** Fix pooled XGBoost (`model.task: regression`) and `configs/default.yaml` splits/labels/`long_short` weekly top-10 / bottom-10 construction. Run `python scripts/run_experiment2.py`. **Stage A** trains one model per family arm (`full`, `returns`, `trend`, `momentum`, `volatility`, `volume`, `returns_volatility`). **Stage B** builds prune arms (`top5`, `top10`, `cum80`) from Full-model train importance and trains them. Best Stage B arm is identified on **val Sharpe**; all arms are reported on frozen test. No calibration, confidence gating, or portfolio-construction changes.
- **Models compared:** Feature arms only — same pooled XGBoost regressor; ranking score = predicted 5-day forward return.
- **Dataset:** Same S&P 100 panel and calendar splits as Experiment 1; last 5 train/val dates purged for label leakage.
- **Features used:** Subsets of the standard 19 technicals, grouped as returns / trend / momentum / volatility / volume (see `src/features/families.py`). Stage B subsets are importance-selected from the Full arm only (never from test).
- **Evaluation methodology:**
  1. Ranking metrics on train/val/test per arm (IC, mean daily IC, IC IR, top-decile hit rate, ROC-AUC, PR-AUC).
  2. Identical long-short backtest on val/test per arm.
  3. Head-to-head vs Full: which family carries signal? does pruning help? differences material? which feature set advances?
- **Success criteria:** A clear, reproducible feature-set recommendation for later experiments based primarily on **test Sharpe** (ties: test mean-daily IC, then top-decile hit rate). Absolute skill may remain modest; the experiment succeeds if it isolates feature-composition effects under a fixed model and strategy.
- **Expected figures/tables for the presentation:**
  - Table: ranking metrics × feature arm × split.
  - Table: trading metrics × feature arm × (val/test).
  - Bars: test Sharpe and mean-daily IC by arm (highlight Full vs best subset).
  - Full-model feature importance chart (Stage B selection rationale).
  - Optional: feature correlation heatmap of the 19 technicals.

### Experiment 3 — Calibration Quality

- **Objective:** Determine whether raw probabilities from the winning model/feature configuration of earlier experiments need post-hoc calibration, and whether calibration improves probability quality without disturbing rank order.
- **Hypothesis tested:** H3.
- **Experimental setup:** Using the chosen model and feature set from Experiments 1–2, run `python main.py calibrate`, which executes `run_calibration_analysis()`: fits Platt and isotonic calibrators on validation raw scores, then evaluates classification metrics (Brier, ECE, reliability bins) for raw/Platt/isotonic probabilities on both val and test.
- **Models compared:** The three probability variants of the same trained model — raw, Platt-calibrated, isotonic-calibrated.
- **Dataset:** `predictions_val.parquet` and `predictions_test.parquet` from the chosen configuration.
- **Features used:** None directly — this experiment operates on saved probability outputs, not raw features.
- **Evaluation methodology:** Brier score and ECE by probability column × split; reliability diagrams via `plot_calibration_curves` (report figure 09); IC recomputed on each probability column to confirm rank-order invariance under calibration.
- **Success criteria:** A clear, reproducible answer on whether calibration helps, measured against the codebase's own materiality thresholds (ΔBrier > 0.005 or ΔECE > 0.01), with IC essentially unchanged across raw/Platt/isotonic.
- **Expected figures/tables for the presentation:**
  - Reliability diagram: raw vs. Platt vs. isotonic, val and test.
  - Table: Brier score / ECE by probability column × split.
  - Table: IC by probability column (invariance check).
  - Summary callout: `calibrated_probabilities_materially_better` and `best_calibrator_by_val_brier` from `calibration_trading_report_latest.json`.

### Experiment 4 — Confidence-Gated Trading vs. Pure Ranking

- **Objective:** Determine whether gating trades by calibrated probability confidence (`long_short_confidence`) improves realized, out-of-sample risk-adjusted returns relative to pure top/bottom-N ranking (`long_short`).
- **Hypothesis tested:** H4.
- **Experimental setup:** Using the same `run_calibration_analysis()` run from Experiment 3, compare its Strategy A (`long_short`, pure ranking) against Strategy B (`long_short_confidence`) swept over the 3 threshold pairs `{(0.60,0.40), (0.65,0.35), (0.70,0.30)}`, selected on validation by Sharpe ratio with a 10-trade floor, and evaluated once, frozen, on test.
- **Models compared:** `long_short` vs. `long_short_confidence` portfolio construction modes, built on top of the best-calibrated probability column identified in Experiment 3.
- **Dataset:** `predictions_val.parquet` (threshold selection) and `predictions_test.parquet` (frozen evaluation).
- **Features used:** None directly — this is a portfolio-level backtest on frozen predictions.
- **Evaluation methodology:** `compute_backtest_metrics()` via `run_backtest_on_predictions()` — Sharpe ratio, total trades, win rate, max drawdown, turnover — for Strategy A and for Strategy B at each threshold pair (val sweep) plus the single frozen test result; cross-check against the codebase's own `_build_recommendation()` output.
- **Success criteria:** A clear answer on whether confidence gating improves test Sharpe versus pure ranking while producing at least 10 trades in the validation selection window; the automated `production_mode` recommendation reported and validated against the raw numbers.
- **Expected figures/tables for the presentation:**
  - Table: Strategy A vs. Strategy B backtest metrics at each of the 3 threshold pairs (validation sweep) plus the frozen test result.
  - Bar chart: Sharpe ratio, Strategy A vs. Strategy B (test).
  - Summary callout: `production_mode` recommendation and rationale from `calibration_trading_report_latest.json`.

---

## Recommended Ordering

For a coherent, academic-paper-style narrative, present the experiments in this order:

1. **Experiment 1 — "Which model should drive the ranking strategy?"** (H1). Before anything else can be evaluated, the base model must be chosen; this establishes which pooled tree family produces the strongest weekly rankings **and** the best identical top-10 / bottom-10 long-short portfolio performance on matched data.
2. **Experiment 2 — "Which features should that model use?"** (H2). With the model frozen, this isolates whether feature-family composition or importance pruning improves ranking quality and long-short performance versus the full technical set.
3. **Experiment 3 — "Can we trust its probabilities?"** (H3). Once model and features are chosen, the next question is whether raw probability outputs are well-calibrated or need post-hoc correction before they are used to make trading decisions.
4. **Experiment 4 — "Should we act on confidence?"** (H4). With a calibrated (or confirmed-uncalibrated) probability in hand, this closes the story by testing whether gating trades on that probability's confidence improves realized, out-of-sample risk-adjusted performance versus simply trading the ranks.

This progression — **model → features → probability calibration → trading decision rule** — mirrors the structure of a typical empirical-finance research paper and gives the presentation a clear cause-and-effect throughline: each experiment's output is a direct input to the next.
