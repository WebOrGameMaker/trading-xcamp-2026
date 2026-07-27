# Research Presentation — Pooled XGBoost Cross-Sectional Strategy

This document lays out the research question, hypotheses, and experiments for a presentation on the project's AI-powered cross-sectional equity trading system: a pooled XGBoost model that ranks S&P 100 stocks and drives a weekly long-short portfolio. Everything below is scoped to what the current codebase can actually run — no invented features, models, or data sources.

## Overall Research Question

> Can a pooled, cross-sectional XGBoost classifier trained on standard technical indicators generate statistically robust and economically exploitable rank information across S&P 100 equities out-of-sample, and does the choice of model family, probability calibration, and confidence-based trade gating meaningfully affect the resulting strategy's risk-adjusted (Sharpe) performance?

---

## Hypotheses

### H1 — Model Family and Ranking Skill

| Field                  | Detail                                                                                                                                                                                                    |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Null hypothesis        | ROC-AUC, PR-AUC, and Information Coefficient (IC) on the test split do not differ meaningfully across pooled XGBoost, LightGBM, and Random Forest classifiers under matched features, labels, and splits. |
| Alternative hypothesis | At least one gradient-boosted model (XGBoost or LightGBM) exceeds Random Forest on these metrics.                                                                                                         |
| Independent variables  | `model.type` (`xgboost`, `lightgbm`, `random_forest`) — features, labels, and calendar splits held constant.                                                                                              |
| Dependent variables    | ROC-AUC, PR-AUC, Brier score, IC, top-decile hit rate (test split).                                                                                                                                       |
| Evaluation metrics     | `evaluate_classifier()` outputs and the `evaluate_cross_sectional()` bundle.                                                                                                                              |
| Expected outcome       | A small but directional edge for boosted trees; overall skill is likely modest given the noisy, weak-signal nature of daily technical features.                                                           |
| Codebase support       | `main.py train --model`; `src/models/trainer.py::_build_classifier`; `src/models/evaluator.py`; `src/models/cross_sectional.py`; `tests/test_evaluator.py`; `tests/test_cross_sectional.py`.              |

### H2 — Calibration Quality

| Field                  | Detail                                                                                                                                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Null hypothesis        | Platt scaling and isotonic regression do not reduce Brier score or Expected Calibration Error (ECE) relative to raw model probability on the test split.                                                           |
| Alternative hypothesis | At least one calibrator materially reduces Brier and/or ECE (per the codebase's own materiality thresholds: ΔBrier > 0.005 or ΔECE > 0.01), while rank order (IC) is preserved because the transform is monotonic. |
| Independent variables  | Probability column used: `probability` (raw), `probability_platt`, `probability_isotonic`.                                                                                                                         |
| Dependent variables    | Brier score, ECE, reliability-bin gap, IC (as an invariance check).                                                                                                                                                |
| Evaluation metrics     | `evaluate_classifier()` (`brier_score`, `ece`, `calibration_bins`); `information_coefficient()`.                                                                                                                   |
| Expected outcome       | Tree ensembles tend to be overconfident at the probability extremes; isotonic/Platt calibration should lower Brier/ECE without materially changing IC.                                                             |
| Codebase support       | `src/models/calibration.py`; `src/models/calibration_analysis.py::run_calibration_analysis`; `tests/test_calibration.py`; `tests/test_evaluator.py` (ECE and calibration-bin tests).                               |

### H3 — Confidence-Gated Trading vs. Pure Ranking

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

### Experiment 1 — Model Family and Ranking Skill

- **Objective:** Determine whether the choice of pooled classifier (XGBoost, LightGBM, or Random Forest) materially changes classification quality and cross-sectional ranking skill under otherwise identical features, labels, and splits.
- **Hypothesis tested:** H1.
- **Experimental setup:** Using `configs/default.yaml` (S&P 100 universe, 2010–2022 train / 2023–2024 val / 2025+ test, cross-sectional top-20% 5-day forward-return labels, the 18-feature standard indicator set), train three pooled classifiers via `python main.py train --model {xgboost,lightgbm,random_forest}`, each producing `predictions_{train,val,test}.parquet`.
- **Models compared:** Pooled XGBoost (default hyperparameters), pooled LightGBM (matched hyperparameters), pooled Random Forest (matched depth/estimator budget).
- **Dataset:** S&P 100 (`configs/sp100_tickers.yaml`), daily bars from 2010–present, calendar split per `configs/default.yaml`.
- **Features used:** The 18 standard technical indicators from `get_feature_columns()` — 1/5/20-day returns, price/SMA and price/EMA ratios, MACD (line/signal/histogram, price-normalized), RSI(14), stochastic %K/%D, Bollinger Band width, ATR%, 20-day realized volatility, OBV z-score, and volume/SMA ratio.
- **Evaluation methodology:** For each model, compute classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier) on train/val/test via `evaluate_classifier`; compute the cross-sectional bundle (IC, top-decile hit rate) via `evaluate_cross_sectional` on test; compare the three model types head-to-head on identical data.
- **Success criteria:** A clear, reportable ranking among the three model types on ROC-AUC, PR-AUC, and IC — even if the absolute differences are small — sufficient to recommend a model family for the rest of the pipeline.
- **Expected figures/tables for the presentation:**
  - Table: classification metrics × model type × split.
  - `plot_metric_overview` (report figure 01) per model.
  - `plot_roc_auc_distribution` (report figure 02).
  - Table: IC (overall / mean daily / std daily) and top-decile hit rate by model type.
  - `plot_feature_importance` (report figure 05) for the winning model.

### Experiment 2 — Calibration Quality

- **Objective:** Determine whether raw probabilities from the winning model of Experiment 1 need post-hoc calibration, and whether calibration improves probability quality without disturbing rank order.
- **Hypothesis tested:** H2.
- **Experimental setup:** Using the winning model from Experiment 1, run `python main.py calibrate`, which executes `run_calibration_analysis()`: fits Platt and isotonic calibrators on validation raw scores, then evaluates classification metrics (Brier, ECE, reliability bins) for raw/Platt/isotonic probabilities on both val and test.
- **Models compared:** The three probability variants of the same trained model — raw, Platt-calibrated, isotonic-calibrated.
- **Dataset:** `predictions_val.parquet` and `predictions_test.parquet` from Experiment 1's chosen model.
- **Features used:** None directly — this experiment operates on saved probability outputs, not raw features.
- **Evaluation methodology:** Brier score and ECE by probability column × split; reliability diagrams via `plot_calibration_curves` (report figure 09); IC recomputed on each probability column to confirm rank-order invariance under calibration.
- **Success criteria:** A clear, reproducible answer on whether calibration helps, measured against the codebase's own materiality thresholds (ΔBrier > 0.005 or ΔECE > 0.01), with IC essentially unchanged across raw/Platt/isotonic.
- **Expected figures/tables for the presentation:**
  - Reliability diagram: raw vs. Platt vs. isotonic, val and test.
  - Table: Brier score / ECE by probability column × split.
  - Table: IC by probability column (invariance check).
  - Summary callout: `calibrated_probabilities_materially_better` and `best_calibrator_by_val_brier` from `calibration_trading_report_latest.json`.

### Experiment 3 — Confidence-Gated Trading vs. Pure Ranking

- **Objective:** Determine whether gating trades by calibrated probability confidence (`long_short_confidence`) improves realized, out-of-sample risk-adjusted returns relative to pure top/bottom-N ranking (`long_short`).
- **Hypothesis tested:** H3.
- **Experimental setup:** Using the same `run_calibration_analysis()` run from Experiment 2, compare its Strategy A (`long_short`, pure ranking) against Strategy B (`long_short_confidence`) swept over the 3 threshold pairs `{(0.60,0.40), (0.65,0.35), (0.70,0.30)}`, selected on validation by Sharpe ratio with a 10-trade floor, and evaluated once, frozen, on test.
- **Models compared:** `long_short` vs. `long_short_confidence` portfolio construction modes, built on top of the best-calibrated probability column identified in Experiment 2.
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

1. **Experiment 1 — "Which model should we trust?"** (H1). Before anything else can be evaluated, the base classifier must be chosen; this establishes which pooled model family produces the strongest classification and ranking skill on identical data.
2. **Experiment 2 — "Can we trust its probabilities?"** (H2). Once a model is chosen, the next question is whether its raw probability outputs are well-calibrated or need post-hoc correction before they are used to make trading decisions.
3. **Experiment 3 — "Should we act on confidence?"** (H3). With a calibrated (or confirmed-uncalibrated) probability in hand, this closes the story by testing whether gating trades on that probability's confidence improves realized, out-of-sample risk-adjusted performance versus simply trading the ranks.

This progression — **model selection → probability calibration → trading decision rule** — mirrors the structure of a typical empirical-finance research paper and gives the presentation a clear cause-and-effect throughline: each experiment's output is a direct input to the next.
