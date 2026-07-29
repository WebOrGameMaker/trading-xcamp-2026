# Experiment 1 — Model Family, Ranking Quality, and Strategy Performance

**Status:** Complete. Tests H1 only. Trees predict continuous 5-day forward returns, rank by predicted return, and feed an identical weekly top-10 / bottom-10 long-short pipeline. Calibration and confidence-gated trading are out of scope.

**Run manifest:** [`run_manifest.json`](run_manifest.json) · **Tables:** [`metrics_by_model_split.csv`](metrics_by_model_split.csv), [`cross_sectional_by_model.csv`](cross_sectional_by_model.csv), [`trading_by_model.csv`](trading_by_model.csv), [`returns_by_model.csv`](returns_by_model.csv) · **Per-model artifacts:** `xgboost/`, `lightgbm/`, `random_forest/`, `catboost/`

---

## 1. Methodology

**Universe.** S&P 100 (`configs/sp100_tickers.yaml`).

**Data & splits.** Daily OHLCV, 2010-01-01 → present, calendar split per `configs/default.yaml`:

- Train: ≤ 2022-12-31 (313,337 rows)
- Validation: 2023-01-01 → 2024-12-31 (57,536 rows)
- Test: ≥ 2025-01-01 (44,428 rows)

Last 5 trading dates of train/val purged so the 5-day-forward target cannot leak across split boundaries.

**Target.** Continuous `forward_return_5d` (`model.task: regression`). Binary top-20% labels are retained **only** as a hit-rate evaluation helper, not as the training target.

**Features (19).** Scale-free technical set: returns, price/SMA and price/EMA ratios, MACD (price-normalized), RSI, stochastics, BB width, ATR%, volatility, OBV z-score, volume ratio.

**Models.** Four pooled **regressors** (matched hyperparameters where possible):

| Model | Key hyperparameters |
| --- | --- |
| XGBoost | `n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8` |
| LightGBM | same as above |
| Random Forest | `n_estimators=200, max_depth=10, min_samples_leaf=5` |
| CatBoost | `n_estimators=200, max_depth=6, learning_rate=0.05` |

All use `StandardScaler` → regressor. Ranking score = predicted forward return (stored in the `probability` column for the shared ranking/trading code path).

**Evaluation.** Ranking metrics via `evaluate_cross_sectional`; identical `long_short` weekly backtest via `run_backtest_on_predictions`. Winner ranked by **test Sharpe**, then mean-daily IC, then top-decile hit rate.

---

## 2. Ranking metrics (test)

| Model | IC (overall) | IC (mean daily) | IC IR | Top-decile hit rate | ROC-AUC* |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.022 | 0.000 | 0.001 | 0.293 | 0.518 |
| LightGBM | 0.028 | 0.004 | 0.024 | 0.288 | 0.523 |
| Random Forest | **0.035** | **0.011** | **0.056** | 0.295 | **0.531** |
| CatBoost | 0.029 | 0.006 | 0.033 | **0.299** | 0.526 |

\*ROC-AUC scores predicted return against the binary top-20% label (ranking helper only). Absolute OOS R² is near zero / slightly negative for all models.

---

## 3. Trading performance (identical long-short pipeline)

| Model | Val Sharpe | Test Sharpe | Test ann. return | Test max DD |
| --- | ---: | ---: | ---: | ---: |
| **XGBoost** | 0.90 | **−0.11** | −2.0% | 9.1% |
| LightGBM | 0.68 | −0.11 | −2.1% | 10.3% |
| Random Forest | 0.61 | −0.29 | −5.0% | 14.3% |
| CatBoost | 0.86 | −0.48 | −7.4% | 18.7% |

Every model is profitable on validation and loses on test. Win rates ≈ 47%; profit factors < 1 on test; turnover ≈ 0.6–0.7.

---

## 4. Is H1 supported?

**H1 alternative:** At least one model family is materially better on ranking and/or trading.

**Verdict: Mixed, and not practically useful.**

- Ranking: Random Forest leads mean-daily IC; differences vs XGBoost exceed the ΔIC ≥ 0.005 bar, but absolute skill is tiny (IC IR ≪ 0.1).
- Trading: XGBoost wins by the pre-specified test-Sharpe rule, but only as the **least-negative** OOS result. XGBoost vs LightGBM on test Sharpe is noise (Δ ≈ 0.002). Larger gaps vs RF/CatBoost still leave all strategies losing money.
- Ranking and trading disagree: RF ranks best, XGBoost trades “best.”
- Val→test collapse for all models (all flagged `overfitting`) makes any single-slice winner fragile.

**Recommendation:** Advance **XGBoost** for Experiment 2 because that is the frozen trading-winner rule — not because a profitable edge was found.

---

## 5. Reproducing

```bash
python scripts/run_experiment1.py
python scripts/plot_experiment1.py
```

Requires network only for tickers not already cached in `data/raw/`.
