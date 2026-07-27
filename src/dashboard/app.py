"""Streamlit dashboard for the AI trading bot."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.config import load_config
from src.utils.paths import LOG_DIR, MODEL_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

st.set_page_config(page_title="AI Trading Bot", page_icon="📈", layout="wide")


def _load_json(path: Path) -> dict | None:
    """Load JSON file if it exists."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def render_overview() -> None:
    """Render overview tab with equity curve and key metrics."""
    st.header("Overview")

    metrics = _load_json(LOG_DIR / "backtest_metrics.json")
    equity_path = LOG_DIR / "equity_curve.csv"

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    if metrics:
        col1.metric("Total Return", f"{metrics.get('total_return', 0) * 100:.2f}%")
        col2.metric("Ann. Return", f"{metrics.get('annualized_return', 0) * 100:.2f}%")
        col3.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        col4.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0) * 100:.2f}%")
        col5.metric("Win Rate", f"{metrics.get('win_rate', 0) * 100:.2f}%")
        col6.metric("Turnover", f"{metrics.get('turnover', 0):.2f}")

        bench = metrics.get("benchmark_return", 0)
        st.caption(f"Benchmark (SPY) return: {bench * 100:.2f}%")
    else:
        st.info("Run `python main.py backtest` to generate metrics.")

    if equity_path.exists():
        equity = pd.read_csv(equity_path, index_col=0, parse_dates=True)
        fig = px.line(equity, y="equity", title="Portfolio Equity Curve")
        fig.update_layout(xaxis_title="Date", yaxis_title="Equity ($)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No equity curve data available.")


def render_model_performance() -> None:
    """Render model performance tab."""
    st.header("Model Performance")

    latest_meta = _load_json(MODEL_DIR / "latest.json")
    if not latest_meta:
        st.info("Run `python main.py train` to generate model metrics.")
        return

    all_metrics = latest_meta.get("metrics", {})
    scope = latest_meta.get("scope", "pooled")
    per_symbol = all_metrics.get("per_symbol", {})
    st.caption(
        f"Model scope: {scope} · "
        f"{latest_meta.get('symbol_count', 1)} model(s) · "
        f"run {latest_meta.get('run_id', 'n/a')}"
    )

    if scope == "pooled" or not per_symbol:
        metrics = all_metrics
        selected_symbol = "Aggregate"
    else:
        selected_symbol = st.selectbox(
            "Metrics view",
            ["Aggregate", *sorted(per_symbol)],
        )
        metrics = (
            all_metrics
            if selected_symbol == "Aggregate"
            else per_symbol[selected_symbol]
        )

    columns = st.columns(3)
    gap = metrics.get("generalization_gap")
    if gap and selected_symbol == "Aggregate":
        gap_help = {
            "overfitting": "Train accuracy is notably higher than test — model memorized training noise.",
            "underfitting_or_no_signal": "Both train and test ROC-AUC are near 0.5 — the model isn't finding usable signal.",
            "ok": "Train and test performance are reasonably close.",
            "no_data": "Not enough data to diagnose.",
        }.get(gap, gap)
        st.caption(f"Generalization diagnosis: **{gap}** — {gap_help}")

    for col, split in zip(columns, ("train", "val", "test"), strict=True):
        split_metrics = metrics.get(split, {})
        if split_metrics:
            with col:
                st.subheader(f"{split.title()} Set")
                st.metric("Accuracy", f"{split_metrics.get('accuracy', 0):.3f}")
                st.metric("F1 Score", f"{split_metrics.get('f1', 0):.3f}")
                st.metric("ROC-AUC", f"{split_metrics.get('roc_auc', 0):.3f}")
                st.metric("PR-AUC", f"{split_metrics.get('pr_auc', 0):.3f}")
                st.metric("Precision", f"{split_metrics.get('precision', 0):.3f}")
                st.metric("Recall", f"{split_metrics.get('recall', 0):.3f}")
                st.metric("Brier Score", f"{split_metrics.get('brier_score', 0):.3f}")
                cs = split_metrics.get("cross_sectional", {})
                if cs:
                    st.metric("IC (daily)", f"{cs.get('ic_mean_daily', 0):.4f}")
                    st.metric("Top-decile hit", f"{cs.get('top_decile_hit_rate', 0):.3f}")

    importance = metrics.get("feature_importance", {})
    if importance:
        imp_df = pd.DataFrame(
            {"feature": list(importance.keys()), "importance": list(importance.values())}
        ).head(20)
        fig = px.bar(imp_df, x="importance", y="feature", orientation="h", title="Top 20 Features")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    shap_importance = metrics.get("shap_importance", {})
    if shap_importance:
        shap_df = pd.DataFrame(
            {
                "feature": list(shap_importance.keys()),
                "mean_|SHAP|": list(shap_importance.values()),
            }
        ).head(20)
        fig = px.bar(
            shap_df,
            x="mean_|SHAP|",
            y="feature",
            orientation="h",
            title="Top 20 SHAP Features",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    eval_files = (
        sorted(LOG_DIR.glob(f"eval_*_{selected_symbol}_test.json"))
        if selected_symbol != "Aggregate"
        else sorted(LOG_DIR.glob("eval_*_pooled_test.json"))
    )
    if eval_files:
        eval_data = _load_json(eval_files[-1])
        if eval_data and "confusion_matrix" in eval_data:
            cm = eval_data["confusion_matrix"]
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Pred 0", "Pred 1"],
                y=["True 0", "True 1"],
                colorscale="Blues",
            ))
            fig.update_layout(title="Confusion Matrix (Test)")
            st.plotly_chart(fig, use_container_width=True)


def render_positions() -> None:
    """Render current Alpaca paper positions."""
    st.header("Positions")

    config = load_config()
    if not config.alpaca.api_key or not config.alpaca.secret_key:
        st.warning("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env to view live positions.")
        return

    try:
        from src.execution.alpaca_client import AlpacaClient

        client = AlpacaClient(config.alpaca)
        equity = client.get_account_equity()
        st.metric("Account Equity", f"${equity:,.2f}")

        positions = client.get_positions()
        if positions:
            pos_df = pd.DataFrame([
                {
                    "Symbol": p.symbol,
                    "Qty": p.qty,
                    "Price": p.current_price,
                    "Market Value": p.market_value,
                }
                for p in positions
            ])
            st.dataframe(pos_df, use_container_width=True)
        else:
            st.info("No open positions.")
    except Exception as exc:
        st.error(f"Failed to fetch positions: {exc}")


def render_trade_log() -> None:
    """Render recent trade orders."""
    st.header("Trade Log")

    config = load_config()
    if not config.alpaca.api_key:
        st.warning("Configure Alpaca credentials to view trade log.")
        return

    try:
        from src.execution.alpaca_client import AlpacaClient

        client = AlpacaClient(config.alpaca)
        orders = client.get_recent_orders(limit=50)
        if orders:
            st.dataframe(pd.DataFrame(orders), use_container_width=True)
        else:
            st.info("No recent orders.")
    except Exception as exc:
        st.error(f"Failed to fetch orders: {exc}")


def render_data_health() -> None:
    """Render data pipeline health status."""
    st.header("Data Health")

    raw_files = list(RAW_DATA_DIR.glob("*.parquet")) if RAW_DATA_DIR.exists() else []
    st.metric("Cached Symbols", len(raw_files))

    features_path = PROCESSED_DATA_DIR / "features.parquet"
    if features_path.exists():
        df = pd.read_parquet(features_path)
        st.metric("Feature Rows", f"{len(df):,}")
        st.metric("Symbols in Dataset", df["symbol"].nunique())
        st.metric("Date Range", f"{df['date'].min().date()} → {df['date'].max().date()}")
        label_balance = df["label"].value_counts(normalize=True)
        fig = px.pie(values=label_balance.values, names=label_balance.index.astype(str), title="Label Balance")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python main.py features` to build feature dataset.")

    if raw_files:
        latest = max(f.stat().st_mtime for f in raw_files)
        st.caption(f"Latest raw cache update: {pd.Timestamp(latest, unit='s')}")


def main() -> None:
    """Dashboard entry point."""
    st.title("AI Trading Bot Dashboard")
    st.caption("S&P 100 · pooled cross-sectional XGBoost · Paper trading")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Model Performance",
        "Positions",
        "Trade Log",
        "Data Health",
    ])

    with tab1:
        render_overview()
    with tab2:
        render_model_performance()
    with tab3:
        render_positions()
    with tab4:
        render_trade_log()
    with tab5:
        render_data_health()


if __name__ == "__main__":
    main()
