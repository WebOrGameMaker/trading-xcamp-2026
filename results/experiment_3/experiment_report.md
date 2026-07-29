# Experiment 3 — Target (Label) Engineering

**Status:** Complete. Tests H3 only. Frozen pooled XGBoost from Experiment 1; frozen top-5 feature set from Experiment 2 (`atr_pct`, `return_5d`, `volatility_20d`, `price_sma10_ratio`, `stoch_d`). Same S&P 100 universe, calendar splits, weekly top-10/bottom-10 long-short pipeline, transaction costs, and evaluation code as Experiment 2. The prediction target is the only independent variable.

**Run manifest:** [`run_manifest.json`](run_manifest.json) · **Tables:** [`metrics_by_target_split.csv`](metrics_by_target_split.csv), [`cross_sectional_by_target.csv`](cross_sectional_by_target.csv), [`trading_by_target.csv`](trading_by_target.csv), [`returns_by_target.csv`](returns_by_target.csv), [`deltas_vs_baseline.csv`](deltas_vs_baseline.csv) · **Per-target artifacts:** `A_5d_absolute/`, `B_3d_absolute/`, `C_10d_absolute/`, `D_5d_relative/` · **Figures:** `figures/`

---

## 1. Setup

**Frozen from Experiment 2 (unchanged):**

- Model: pooled XGBoost (`n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`).
- Features (5): `atr_pct`, `return_5d`, `volatility_20d`, `price_sma10_ratio`, `stoch_d` — the Exp 2 `top5` winner.
- Universe: S&P 100 (116 of 117 tickers resolved; `BK` delisted on the data vendor and excluded, same as prior experiments).
- Calendar splits: train ≤ 2022-12-31, val 2023-01-01 → 2024-12-31, test ≥ 2025-01-01.
- Strategy: weekly `long_short`, 10 long / 10 short, equal-weight (±0.50 gross), `max_weight_per_symbol=0.10`.
- Costs: 1.0 bps commission + 5.0 bps slippage = 6 bps/trade; `initial_cash=100,000`.
- Evaluation code: `evaluate_regressor`, `information_coefficient`, `top_decile_hit_rate`, `mean_return_by_prediction_decile`, `run_backtest_on_predictions` — identical implementations to Exp 1/2.

**What changed:** only the regression target `y` fed to XGBoost. A new pooled panel (`data/processed/features_experiment3.parquet`, 415,997 rows) was built alongside — never touching `features.parquet` — carrying `forward_return_3d`, `forward_return_5d`, `forward_return_10d`, `forward_return_5d_rel`, and a binary `label` fixed at the 5-day horizon, all computed on the same technical features and the same raw OHLCV history.

**Fairness rule (locked before running):** because a shorter or longer horizon is intrinsically easier or harder to fit, every target's **ranking metrics** (IC, IC IR, hit rate, decile spread) are scored against one **common yardstick** — the absolute `forward_return_5d` column and the label fixed at 5 days — never against each model's own training horizon. Own-target RMSE/MAE/R² are reported separately as fit diagnostics only. Trading metrics use the identical weekly long/short backtest for all four targets, since portfolio construction depends only on rank order.

**Minor documented deviation from Exp 1/2's row counts:** the shared multi-target panel requires all four target columns to be simultaneously non-null on every retained row (so all four arms train/evaluate on an identical row set). This trims a small number of dates from the very end of the sample (where the 10-day forward return isn't yet realized) that Exp 1/2's single-target panel did not need to drop. Effect: train/val row counts for the 5-day arms match Exp 1/2 exactly (313,337 train / 57,536 val); test shrinks from 44,428 to 43,848 rows (−1.3%) for every arm equally. This does not bias any target relative to another.

## 2. Target definitions

| Target | Column | Purge (trading days) | Definition |
| --- | --- | --- | --- |
| **A (baseline)** | `forward_return_5d` | 5 | `close[t+5] / close[t] − 1` |
| **B** | `forward_return_3d` | 3 | `close[t+3] / close[t] − 1` |
| **C** | `forward_return_10d` | 10 | `close[t+10] / close[t] − 1` |
| **D** | `forward_return_5d_rel` | 5 | `forward_return_5d − median_j(forward_return_5d)` within each date |

Pre-implementation review (quant-researcher pass, done before running): the four targets form a coherent 2-factor design — a horizon sweep (B/C bracket A) plus a loss-geometry change (D changes what XGBoost optimizes for, not the horizon). D was kept rather than substituted: for a dollar-neutral top-10/bottom-10 book, the common market/day component in an absolute return cancels in P&L anyway, so training on the demeaned target should in principle force the model to fit only the part of the return that separates winners from losers. No target was replaced.

## 3. Trading performance (identical long-short pipeline)

| Target | Val Sharpe | Test Sharpe | Test ann. return | Test max DD | Test win rate | Test profit factor | Test turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **A: 5d abs (baseline)** | 0.09 | **0.64** | **+7.2%** | 12.1% | 49.5% | 1.08 | 0.73 |
| B: 3d abs | −0.50 | 0.06 | +0.1% | 10.9% | 47.4% | 1.00 | 0.73 |
| C: 10d abs | **0.88** | 0.56 | +6.3% | **5.4%*** | 49.7% | 1.08 | 0.69 |
| D: 5d relative | −0.63 | −0.32 | −4.3% | 19.9% | 47.5% | 0.95 | 0.71 |

\*C's max drawdown of 5.4% is the *validation*-window figure; C's test-window max drawdown is 11.5%, similar to A.

Full detail: [`trading_by_target.csv`](trading_by_target.csv) · Figures: `figures/target_comparison_trading_test.png`, `figures/target_comparison_trading_val.png`.

Ranking by test Sharpe: **A > C > B > D**. Ranking by val Sharpe: **C > A > B > D** — C is the best-by-validation arm, consistent with a genuinely smoother, more predictable 10-day signal, but its test Sharpe (0.56) is still below A's (0.64).

## 4. Ranking performance (common 5-day yardstick)

| Target | IC (overall) | IC (mean daily, test) | IC IR (test) | Top-decile hit rate (test) | Top−bottom decile spread (test) | ROC-AUC (test) | PR-AUC (test) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: 5d abs (baseline) | 0.028 | 0.0149 | 0.088 | 0.295 | 0.0050 | 0.537 | 0.246 |
| B: 3d abs | 0.030 | 0.0171 | 0.103 | 0.294 | 0.0045 | 0.533 | 0.245 |
| **C: 10d abs** | **0.034** | **0.0220** | **0.127** | **0.304** | **0.0069** | 0.542 | 0.250 |
| D: 5d relative | 0.007 | 0.0019 | 0.014 | 0.293 | 0.0014 | 0.521 | 0.237 |

Full detail: [`cross_sectional_by_target.csv`](cross_sectional_by_target.csv), [`returns_by_target.csv`](returns_by_target.csv), [`metrics_by_target_split.csv`](metrics_by_target_split.csv). Figures: `figures/target_comparison_cross_sectional_test.png`, `figures/target_decile_spread_test.png`, `figures/feature_importance_by_target.png`.

**C is the clear ranking-quality winner** on every ranking metric (IC, IC IR, hit rate, decile spread, ROC/PR-AUC), all measured against the same absolute-5-day yardstick as every other arm. **D is the clear ranking-quality loser** — cross-sectional demeaning erased most of the usable signal rather than concentrating it.

Feature importance shifts modestly by target ([`figures/feature_importance_by_target.png`](figures/feature_importance_by_target.png)): `atr_pct` dominates more at the 10-day horizon (0.30 vs 0.22–0.23 for A/D), while `return_5d` and `stoch_d` gain relative weight for the relative-return target D. The ranking of the 5 features never fully reorders, so the frozen top-5 feature set is not obviously miscalibrated for any target.

## 5. Findings

1. **No target beats the baseline on trading performance.** A's test Sharpe (0.64) is the highest of the four; every alternative is flat-to-negative or worse.
2. **C (10-day) is the strongest ranking signal but the second-best trader.** Its IC advantage (ΔIC = +0.0071, above the 0.005 materiality bar) does not convert into a materially better Sharpe or return — a rank/trading disagreement, echoing the same pattern seen with Random Forest in Experiment 1.
3. **B (3-day) is materially worse on trading despite a slightly higher IC than baseline.** Its ΔIC (+0.0022) is below the materiality bar, but its ΔSharpe (−0.57) and ΔAnnRet (−7.1pp) are large negative moves — most likely because a 3-day label is noisier and the weekly rebalance cadence does not match a 3-day holding horizon, adding turnover-adjacent noise without adding tradable signal.
4. **D (cross-sectional relative return) underperforms on every axis.** This is the opposite of the pre-registered expectation that demeaning would sharpen the ranking signal. A plausible explanation: median-demeaning removes not just a common "beta" component but also some of the genuinely reusable cross-sectional structure that the top-5 features (mostly volatility/momentum, not clean value/quality factors) actually capture, leaving mostly idiosyncratic noise for XGBoost to fit.
5. **All three non-baseline deltas are "material" by the pre-registered thresholds** (ΔSharpe ≥ 0.10 or ΔAnnRet ≥ 2pp or ΔIC ≥ 0.005) — but two of the three (B, D) are material in the *wrong* direction, and the third (C) is material only on ranking quality, not trading. See [`deltas_vs_baseline.csv`](deltas_vs_baseline.csv) and `figures/delta_vs_baseline.png`.

## 6. Decision for future pipeline

**Keep the 5-day absolute forward return as the production target.** No alternative target tested here produces a materially better out-of-sample Sharpe or return; the only material ranking-quality improvement (C, 10-day) does not carry through to trading and would also require re-timing the rebalance cadence (a 10-day signal traded on a 5-trading-day/weekly cycle is a mismatch) — out of scope for "target only" changes. The 10-day horizon is worth a dedicated follow-up if rebalance frequency is ever revisited, but is not a drop-in replacement today.

## 7. H3 assessment

**H3 (alternative):** Changing the prediction target produces significantly better out-of-sample ranking quality **and** trading performance than the 5-day absolute baseline.

**Verdict: Not supported.**

- Ranking quality: one target (C) shows a material, consistent improvement across every ranking metric.
- Trading performance: no target shows a material improvement; two (B, D) show large material degradation, and the ranking-quality winner (C) is trading-neutral-to-slightly-worse.
- H3 requires both legs to hold. Since trading performance never materially improves, the null hypothesis (target changes do not materially improve ranking **and** portfolio performance jointly) is retained.

## 8. Risks and limitations

- **Single test window.** All test-split conclusions rest on ~80 weekly rebalances (2025+); none of these Sharpe differences would survive a formal significance test at this sample size.
- **Val/test disagreement.** C wins on validation Sharpe (0.88) but is second on test Sharpe (0.56); A is the reverse (weak on val, strongest on test). This instability was already flagged in Experiment 2 for the same top-5 feature set and persists here — target choice does not fix it.
- **B and D purge lengths differ from A/C**, which very slightly changes which trading dates are available near split boundaries; this is a required, symmetric side effect of matching each target's own horizon and does not favor any arm.
- **D's demeaning is same-day/cross-sectional, not same-week.** The median is computed over each individual trading date, not per rebalance date; a rebalance-date-only demeaning was considered and rejected as a larger, less-comparable design change (see Section 2 rationale carried over from the pre-implementation review).
- **Feature set was not re-tuned per target.** The frozen top-5 features were selected by Experiment 2 for the 5-day absolute target; a different target could in principle prefer a different feature subset. This is out of scope by design ("only the prediction target changes") but means C's and D's results are lower bounds/upper bounds on what a fully re-optimized pipeline could achieve for those targets.

## 9. Bottom-line conclusion

Target engineering moves the needle **more** than the model-family swap in Experiment 1 did (Sharpe spread 0.95 vs 0.37; IC spread 0.020 vs 0.011), which answers success criterion 4 — but nearly all of that extra movement is in the *wrong* direction. Of four targets tested, the current production target (absolute 5-day forward return) remains the best out-of-sample trader; the only target with a materially better ranking signal (10-day absolute) does not convert to better trading under the frozen weekly long/short pipeline, and cross-sectional demeaning (5-day relative) is the worst performer on every metric. **Recommendation: do not switch the production target.** If a future experiment revisits rebalance cadence, the 10-day target's ranking-quality edge is worth re-testing under a matched (bi-weekly or 10-day) rebalance schedule rather than the current weekly one.

---

## Reproducing

```bash
python scripts/run_experiment3.py
```

Requires the existing raw OHLCV cache under `data/raw/` (no new download needed if Experiment 1/2 have already been run) and the Experiment 2 artifact `results/experiment_2/top5/feature_columns.json`. Writes `data/processed/features_experiment3.parquet` and all comparison tables/figures under `results/experiment_3/`, without modifying `data/processed/features.parquet` or any Experiment 1/2 artifact.
