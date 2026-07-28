# Experiment 1 — Model Family and Ranking Skill (Continuous Returns)

**Status:** Complete (re-run). Tests H1 only. Trees predict continuous 5-day forward returns and rank by predicted return. Calibration and confidence-gated trading remain out of scope.

**Run manifest:** `[run_manifest.json](run_manifest.json)` · **Raw tables:** `[metrics_by_model_split.csv](metrics_by_model_split.csv)`, `[cross_sectional_by_model.csv](cross_sectional_by_model.csv)`, `[returns_by_model.csv](returns_by_model.csv)` · **Per-model artifacts:** `xgboost/`, `lightgbm/`, `random_forest/`

---

## 1. Methodology

**Universe.** S&P 100 (`configs/sp100_tickers.yaml`). 116 downloaded successfully; `BK` failed (Yahoo delist/history issue) and was excluded.

**Data & splits.** Daily OHLCV, 2010-01-01 → present, calendar split per `configs/default.yaml`:

- Train: ≤ 2022-12-31 (313,337 rows)
- Validation: 2023-01-01 → 2024-12-31 (57,536 rows)
- Test: ≥ 2025-01-01 (44,428 rows)

Last 5 trading dates of train/val purged so the 5-day-forward target cannot leak across split boundaries.

**Target.** Continuous `forward_return_5d` (`model.task: regression`). Binary top-20% labels are retained **only** as a hit-rate evaluation helper, not as the training target.

**Features (19).** Same scale-free technical set as before (returns, price ratios, MACD, RSI, stochastics, BB width, ATR%, volatility, OBV z-score, volume ratio).

**Models.** Three pooled **regressors** (matched hyperparameters where possible):

| Model | Key hyperparameters |
| --- | --- |
| XGBoost | `n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8` |
| LightGBM | same as above |
| Random Forest | `n_estimators=200, max_depth=10, min_samples_leaf=5` |

All use `StandardScaler` → regressor. Ranking score = predicted forward return.

**Evaluation.** RMSE / MAE / R² via `evaluate_regressor`; cross-sectional IC, top-decile hit rate, and mean return by prediction decile via `evaluate_cross_sectional`. Winner ranked by test IC (mean daily), then hit rate, then R².

---

## 2. Metrics — regression fit, by model × split (test highlight)


| Model | Split | RMSE | MAE | R² | ROC-AUC* | PR-AUC* |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost | **test** | 0.0476 | 0.0336 | −0.0099 | 0.518 | 0.240 |
| LightGBM | **test** | 0.0476 | 0.0336 | −0.0090 | 0.523 | 0.240 |
| Random Forest | **test** | 0.0474 | 0.0335 | −0.0038 | 0.531 | 0.243 |

\*ROC-AUC / PR-AUC here score predicted return against the binary top-20% label (ranking helper only). Absolute R² is near zero / slightly negative OOS — expected for noisy daily equity returns; ranking metrics matter more for this strategy.

---

## 3. Metrics — cross-sectional ranking (test)


| Model | IC (overall) | IC (mean daily) | IC (std daily) | Top-decile hit rate |
| --- | --- | --- | --- | --- |
| XGBoost | 0.022 | 0.000 | 0.182 | 0.293 |
| LightGBM | 0.028 | 0.004 | 0.186 | 0.288 |
| Random Forest | **0.035** | **0.011** | 0.200 | **0.295** |

---

## 4. Returns comparison (test)


| Model | Top-decile mean return | Bottom-decile mean return | Top − bottom |
| --- | --- | --- | --- |
| XGBoost | 0.0104 | 0.0066 | 0.0038 |
| LightGBM | 0.0099 | 0.0061 | 0.0038 |
| Random Forest | **0.0107** | 0.0031 | **0.0077** |

Figures: `figures/model_comparison_returns.png`, `figures/model_comparison_prediction_deciles.png`, `figures/model_comparison_cross_sectional.png`.

---

## 5. Is H1 supported?

**H1:** At least one gradient-boosted model exceeds Random Forest on ranking-quality metrics.

**Verdict: H1 is not supported.** Random Forest leads on test IC (mean daily 0.011), overall IC (0.035), top-decile hit rate (0.295), and top−bottom return spread (0.0077). LightGBM is second on IC; XGBoost is weakest. Boosted trees do not beat RF under these matched/untuned settings when predicting continuous returns.

Absolute skill is modest (mean daily IC ≈ 0–0.01; hit rate ≈ 0.29 vs ~0.20 base rate). Train→test R² collapse (≈0.10 → ≤0) flags overfitting / regime shift; ranking metrics degrade less severely than fit R².

**Recommendation:** Advance **Random Forest** for later experiments (soft call vs LightGBM on some secondary metrics).

---

## 6. Reproducing

```
python scripts/run_experiment1.py
```

Requires network only for tickers not already cached in `data/raw/`.
