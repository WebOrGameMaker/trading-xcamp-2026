# Experiment 2 — Feature Family Ablation and Selection

**Status:** Complete. Tests H2 only. Frozen model = pooled XGBoost from Experiment 1. Same weekly top-10 / bottom-10 long-short pipeline for every feature arm. Calibration and confidence gating are out of scope.

**Run manifest:** [`run_manifest.json`](run_manifest.json) · **Tables:** [`metrics_by_arm_split.csv`](metrics_by_arm_split.csv), [`cross_sectional_by_arm.csv`](cross_sectional_by_arm.csv), [`trading_by_arm.csv`](trading_by_arm.csv), [`returns_by_arm.csv`](returns_by_arm.csv) · **Per-arm artifacts:** `full/`, `returns/`, `trend/`, `momentum/`, `volatility/`, `volume/`, `returns_volatility/`, `top5/`, `top10/`, `cum80/`

---

## 1. Methodology

**Frozen setup.** Pooled XGBoost regressor; continuous `forward_return_5d`; S&P 100 calendar splits (train ≤ 2022 / val 2023–2024 / test ≥ 2025); weekly `long_short` 10 long / 10 short.

**Stage A — family ablation**

| Arm | Features |
| --- | --- |
| `full` | All 19 technicals (baseline) |
| `returns` | `return_1d`, `return_5d`, `return_20d` |
| `trend` | SMA/EMA ratios + MACD |
| `momentum` | RSI + stochastics |
| `volatility` | BB width, ATR%, 20d vol |
| `volume` | OBV z-score, volume/SMA ratio |
| `returns_volatility` | returns ∪ volatility |

**Stage B — importance prune (from Full train importance only)**

| Arm | Rule |
| --- | --- |
| `top5` | Top 5 features by importance |
| `top10` | Top 10 features |
| `cum80` | Smallest set covering ≥ 80% normalized importance |

**Winner rules.** Overall winner: **test Sharpe** → mean-daily IC → top-decile hit rate. Separately recorded: best Stage B arm by **val Sharpe**.

---

## 2. Ranking metrics (test)

| Arm | Stage | n | IC (mean daily) | IC IR | Hit rate |
| --- | --- | ---: | ---: | ---: | ---: |
| full | A | 19 | 0.000 | 0.001 | 0.293 |
| returns | A | 3 | **0.014** | **0.091** | 0.273 |
| trend | A | 8 | −0.001 | −0.006 | 0.271 |
| momentum | A | 3 | −0.012 | −0.073 | 0.216 |
| volatility | A | 3 | −0.001 | −0.009 | 0.294 |
| volume | A | 2 | −0.008 | −0.049 | 0.196 |
| returns_volatility | A | 6 | 0.013 | 0.073 | **0.300** |
| top5 | B | 5 | 0.014 | 0.083 | 0.294 |
| top10 | B | 10 | 0.006 | 0.032 | 0.297 |
| cum80 | B | 15 | −0.001 | −0.004 | 0.295 |

---

## 3. Trading performance (identical long-short pipeline)

| Arm | Val Sharpe | Test Sharpe | Test ann. return |
| --- | ---: | ---: | ---: |
| full | 0.90 | −0.11 | −2.0% |
| returns | 0.94 | −0.57 | −7.6% |
| trend | 0.06 | −1.04 | −11.4% |
| momentum | −0.66 | −0.57 | −6.0% |
| volatility | 0.75 | −0.15 | −2.4% |
| volume | −0.24 | −1.40 | −13.7% |
| returns_volatility | **1.21** | 0.04 | −0.4% |
| **top5** | 0.09 | **0.56** | **+6.2%** |
| top10 | 0.45 | 0.11 | +0.6% |
| cum80 | 0.75 | 0.20 | +1.8% |

Declared overall winner: **top5**. Best Stage B by val Sharpe: **cum80** (not top5).

Full-model top importances: `atr_pct`, `return_5d`, `volatility_20d`, `price_sma10_ratio`, `stoch_d` (importance mass is relatively flat across features).

---

## 4. Is H2 supported?

**H2 alternative:** At least one feature subset materially beats Full on test Sharpe and/or mean-daily IC.

**Verdict: Supported on differences vs Full; not supported as a durable production feature set.**

- Several arms beat Full by ΔSharpe ≫ 0.10 (e.g. top5, cum80, returns_volatility).
- Returns and volatility carry the usable signal; momentum/volume/trend-alone look like drag.
- **top5** wins test Sharpe but fails the Stage B val-selection story (val Sharpe ≈ 0.09 → test 0.56). That inversion makes the declared winner fragile on a short 2025+ window.
- **returns_volatility** is the most coherent Stage A recommendation: best validation Sharpe and roughly flat OOS trading.

**Recommendation:** Prefer **returns + volatility** as the interpretable advance from Exp 2. Treat **top5** as an interesting but not val-validated candidate. Do not claim a robust weekly alpha from this run alone.

---

## 5. Reproducing

```bash
python scripts/run_experiment2.py
python scripts/plot_experiment2.py
```

Requires an existing feature dataset (or network for any uncached tickers). Regenerates val and test comparison figures under `figures/`.
