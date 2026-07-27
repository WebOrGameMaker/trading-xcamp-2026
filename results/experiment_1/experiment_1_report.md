# Experiment 1 — Model Family and Ranking Skill

**Status:** Complete. Tests H1 only. Calibration (Experiment 2) and confidence-gated trading (Experiment 3) are explicitly out of scope for this report.

**Run manifest:** `[run_manifest.json](run_manifest.json)` · **Raw tables:** `[metrics_by_model_split.csv](metrics_by_model_split.csv)`, `[cross_sectional_by_model.csv](cross_sectional_by_model.csv)` · **Per-model artifacts:** `xgboost/`, `lightgbm/`, `random_forest/` (predictions, full manifest, figures)

---

## 1. Methodology

**Universe.** S&P 100 (`configs/sp100_tickers.yaml`, 117 tickers). 116 downloaded successfully via Yahoo Finance; `BK` failed (Yahoo returned "possibly delisted" for the requested history window — a data-availability issue, not a pipeline bug) and was excluded.

**Data & splits.** Daily OHLCV, 2010-01-01 → present, calendar split per `configs/default.yaml`:

- Train: ≤ 2022-12-31 (313,337 rows)
- Validation: 2023-01-01 → 2024-12-31 (57,536 rows)
- Test: ≥ 2025-01-01, fully out-of-sample (44,428 rows)

The last 5 trading dates of the train and validation pools are purged so the 5-day-forward label can never leak across a split boundary.

**Label.** Cross-sectional binary classification: on each date, the top 20% of stocks by 5-day forward return are labeled 1, the rest 0 (`labels.mode: cross_sectional`, `positive_quantile: 0.2`). This is a *ranking* label, not an absolute up/down label — it is designed to reward models that rank stocks well within a date, not models that predict raw direction.

**Features (19, ticker-identity excluded).** `return_{1,5,20}d`, `price_{sma10,sma20,sma50,ema12,ema26}_ratio`, `macd_pct`, `macd_signal_pct`, `macd_hist_pct`, `rsi_14`, `stoch_k`, `stoch_d`, `bb_width`, `atr_pct`, `volatility_20d`, `obv_zscore_60`, `volume_sma_ratio`. All are scale-free/stationary so a single pooled model can compare AAPL and a $20 stock on equal footing.

**Models.** Three *pooled* classifiers (one model trained on all 116 symbols stacked together, no per-ticker models), matched as  bclosely as each library allows via `configs/default.yaml`:


| Model         | Key hyperparameters                                                                      |
| ------------- | ---------------------------------------------------------------------------------------- |
| XGBoost       | `n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8` |
| LightGBM      | same as above                                                                            |
| Random Forest | `n_estimators=200, max_depth=10, min_samples_leaf=5, class_weight=balanced`              |


All three use `StandardScaler` → classifier in an `sklearn.Pipeline`, and `scale_pos_weight` (or `class_weight="balanced"`) set from the train split's class ratio (≈3.9:1 negative:positive) since the label is imbalanced by construction.

**Evaluation.** For each model, `evaluate_classifier()` computes accuracy/precision/recall/F1/ROC-AUC/PR-AUC/Brier/ECE on the *entire* pooled train/val/test split (not per-symbol), and `evaluate_cross_sectional()` computes Spearman IC (overall and mean-daily) and top-decile hit rate on the test split. Predictions, feature importance, SHAP importance, and full metrics are archived per model under `results/experiment_1/{model}/`.

**A bug found and fixed during this run.** `compute_shap_importance()` in `src/models/evaluator.py` crashed for `RandomForestClassifier` because current `shap` returns a 3D `(samples, features, classes)` array for that explainer instead of the list-of-arrays format XGBoost/LightGBM produce. Fixed by selecting the positive-class slice when the SHAP output is 3D; all 82 pre-existing tests still pass after the fix.

**Out of scope (by instruction).** `train_model()` also fits Platt/isotonic calibrators on validation and logs a calibration comparison as an existing side effect of the trainer. Those artifacts exist in `logs/calibration_comparison_*.json` but are **not** analyzed here — that is Experiment 2.

---



## 2. Metrics — classification, by model × split


| Model         | Split    | Accuracy | Precision | Recall | F1    | ROC-AUC   | PR-AUC | Brier | ECE   |
| ------------- | -------- | -------- | --------- | ------ | ----- | --------- | ------ | ----- | ----- |
| XGBoost       | train    | 0.607    | 0.281     | 0.590  | 0.381 | 0.647     | 0.336  | 0.235 | 0.282 |
| XGBoost       | val      | 0.560    | 0.252     | 0.574  | 0.351 | 0.591     | 0.271  | 0.245 | 0.288 |
| XGBoost       | **test** | 0.430    | 0.226     | 0.725  | 0.345 | **0.574** | 0.260  | 0.264 | 0.319 |
| LightGBM      | train    | 0.587    | 0.267     | 0.583  | 0.366 | 0.625     | 0.310  | 0.238 | 0.284 |
| LightGBM      | val      | 0.550    | 0.251     | 0.592  | 0.353 | 0.595     | 0.273  | 0.245 | 0.290 |
| LightGBM      | **test** | 0.418    | 0.227     | 0.755  | 0.349 | **0.584** | 0.269  | 0.265 | 0.321 |
| Random Forest | train    | 0.615    | 0.287     | 0.590  | 0.386 | 0.645     | 0.346  | 0.236 | 0.285 |
| Random Forest | val      | 0.574    | 0.258     | 0.568  | 0.355 | 0.598     | 0.274  | 0.245 | 0.289 |
| Random Forest | **test** | 0.435    | 0.230     | 0.735  | 0.350 | **0.586** | 0.268  | 0.262 | 0.318 |


**Read this carefully:** test accuracy (~0.42–0.44) looks *worse* than the 80% you'd get from always predicting "0" (since only 20% of rows are labeled 1 by construction). That's an artifact of evaluating a `scale_pos_weight`-rebalanced classifier at a raw 0.5 threshold — the confusion matrices below show all three models over-predict class 1 on the test split (Random Forest: 64% of true-0 test rows are predicted 1). Accuracy/precision/recall at 0.5 are **not** the right lens for this task. **ROC-AUC, PR-AUC, and IC are threshold-independent and are what actually matter** for a ranking-based strategy — on those, all three models sit modestly but consistently above chance (AUC 0.57–0.59, PR-AUC ~0.26–0.27 vs. a 0.20 no-skill baseline).

Classification metrics — Random Forest
*(Same figure for [XGBoost](xgboost/figures/01_classification_metric_overview.png) and [LightGBM](lightgbm/figures/01_classification_metric_overview.png).)*

Confusion matrices — Random Forest

**Note on** `plot_roc_auc_distribution`**:** because this is a single pooled model (not one model per ticker), the per-symbol field in the saved eval reports collapses to one entry (`"pooled"`), so the "distribution" figure below is a single point per split rather than a box plot across 116 tickers. It is still generated correctly by the existing visualization code, just not informative as a spread — [xgboost](xgboost/figures/02_roc_auc_distribution.png), [lightgbm](lightgbm/figures/02_roc_auc_distribution.png), [random_forest](random_forest/figures/02_roc_auc_distribution.png).

---



## 3. Metrics — cross-sectional ranking (test split)


| Model         | IC (overall) | IC (mean daily) | IC (std daily) | Top-decile hit rate |
| ------------- | ------------ | --------------- | -------------- | ------------------- |
| XGBoost       | 0.038        | 0.025           | 0.217          | 0.318               |
| LightGBM      | **0.044**    | 0.033           | 0.223          | **0.335**           |
| Random Forest | 0.043        | **0.036**       | 0.235          | 0.334               |


Top-decile hit rate is the fraction of names in the model's top 10% by predicted probability, on a given date, that actually landed in the true top-20% bucket. A hit rate of ~0.33 against a ~0.20 no-skill baseline (roughly matching the 20% positive rate) indicates modest but real ranking skill for all three models. Daily IC of 0.025–0.036 is small in absolute terms but consistent with typical single-signal equity ranking models, and directionally positive out of sample for all three.

Model comparison — test ROC-AUC / PR-AUC / IC

---



## 4. Feature importance (winning model: Random Forest)

Feature importance — Random Forest

The three models agree almost exactly on what matters: `atr_pct` **(normalized ATR) is the single dominant feature in all three** (XGBoost gain 0.17, LightGBM split-count rank #1, Random Forest Gini 0.27), followed by `volatility_20d` (#2 in all three) and `bb_width` (#3 in all three). SHAP importance tells the same story for XGBoost and LightGBM (`atr_pct` far ahead of everything else). In other words: **the pooled models are mostly learning "how volatile is this stock right now," not a genuine directional/momentum alpha signal.** Return features (`return_1d/5d/20d`) and oscillators (RSI, MACD, stochastics) all contribute, but far less than the volatility cluster. This cross-model agreement is reassuring (the signal isn't a model-specific artifact) but also flags a real limitation: the current feature set may be under-representing genuine predictive signal relative to volatility/regime information.

---



## 5. Strengths and weaknesses by model

**XGBoost**

- Weakest of the three on every single test metric (ROC-AUC, PR-AUC, IC-overall, IC-mean-daily, hit rate) at these matched, untuned hyperparameters.
- Largest train→test ROC-AUC drop (0.647 → 0.574, −0.073), suggesting it fits the train-period volatility regime somewhat more tightly than it generalizes.
- Still clearly above chance out of sample (AUC 0.574, IC 0.025), so it retains real signal — just the least of the three here.

**LightGBM**

- Best (or tied-best) on PR-AUC (0.269), IC-overall (0.044), and top-decile hit rate (0.335) — the metrics most directly tied to "does the top of the ranking actually contain the winners."
- Second-best on ROC-AUC (0.584) and IC-mean-daily (0.033), essentially tied with Random Forest (gaps of 0.002–0.003, within noise for a ~500-trading-day test window).
- Fastest to train of the three (~2–3s for a 313k-row fit) — a practical advantage for iteration speed and future walk-forward/backtest loops.

**Random Forest**

- Best on ROC-AUC (0.586) and IC-mean-daily (0.036), i.e. the primary metrics named in the research doc.
- Highest test PR-AUC among the boosted-vs-bagged comparison is a virtual tie with LightGBM (0.268 vs 0.269).
- Slowest to train (~12s, dominated by 200 deep trees at `n_jobs=-1`) and slowest for SHAP (had to be patched to run at all against this sklearn/shap version combination).
- IC std-daily is highest (0.235 vs 0.217–0.223), meaning its daily ranking quality is also the noisiest of the three — a slightly less stable signal day to day despite the best average.

**All three** are flagged `"overfitting"` by the trainer's crude accuracy-gap heuristic (train accuracy − test accuracy > 0.05). Treat that label with caution here: it's measuring raw 0.5-threshold accuracy under a rebalanced classifier, which is dominated by the confusion-matrix skew described in §2, not necessarily by classic overfitting. The threshold-independent ROC-AUC gap (0.06–0.09 train→test) is a fairer read and shows real but moderate degradation — consistent with a mix of some overfitting and a genuine regime shift between the 2010–2022 train window and the 2025+ test window, rather than the model memorizing noise.

---



## 6. Is H1 supported?

**H1 (from the research doc):** *H0 — ROC-AUC, PR-AUC, and IC do not differ meaningfully across pooled XGBoost, LightGBM, and Random Forest classifiers. H1 — at least one gradient-boosted model (XGBoost or LightGBM) exceeds Random Forest on ranking-quality metrics.*

**Verdict: H1 is not supported by this run.** The gradient-boosted models do not consistently beat Random Forest:

- **Random Forest wins** on ROC-AUC (0.586 vs. 0.584/0.574) and IC-mean-daily (0.036 vs. 0.033/0.025) — the two most standard "ranking quality" metrics.
- **LightGBM wins** on PR-AUC, IC-overall, and top-decile hit rate — but only by 0.001–0.002, well within what a ~500-trading-day, single-run test period can produce as noise.
- **XGBoost is strictly worst** on every metric, so if anything the more sophisticated boosting implementation (LightGBM) only ties with bagging, and the other boosting implementation (XGBoost) underperforms it.

So the honest reading is: **model family does not have a large, consistent effect here at matched/untuned hyperparameters** — all three sit in a tight band (ROC-AUC 0.574–0.586, IC-mean-daily 0.025–0.036), and the ranking between LightGBM and Random Forest flips depending on which metric you emphasize. This is closer to the null hypothesis than the alternative. It does **not** mean the research question fails, though: all three models show small but real, directionally positive out-of-sample ranking skill (AUC > 0.5, IC > 0, hit rate > base rate on genuinely held-out 2025+ data), which is the more fundamental thing Experiment 1 needed to establish before Experiments 2–3 are worth running.

---



## 7. Recommendation for Experiment 2

**Advance Random Forest.** Rationale:

1. It leads on the research doc's primary metrics (ROC-AUC, IC-mean-daily).
2. LightGBM is a very close second (differences on its best metrics are ≤0.002, i.e. within noise) and would be a reasonable alternate choice — flag this as a soft call, not a decisive one.
3. Random Forest's `class_weight="balanced"` probabilities and the calibration-comparison artifacts already generated during this run (`logs/calibration_comparison_*.json`, not analyzed here) give Experiment 2 a ready starting point.

If Experiment 2's calibration analysis surfaces meaningfully worse calibration behavior for Random Forest vs. LightGBM (e.g. via ECE or reliability curves), that would be a legitimate reason to reconsider and use LightGBM instead — the two are close enough on ranking skill alone that calibration quality is a fair tie-breaker for a strategy that (in Experiment 2/3) intends to gate trades on predicted confidence.

---



## 8. Reproducing this run

```
python scripts/run_experiment1.py
```

Regenerates `data/processed/features.parquet` from the current codebase, trains all three pooled models fresh (new `run_id`s), and rewrites everything under `results/experiment_1/`. Requires network access for any tickers not already cached in `data/raw/`.