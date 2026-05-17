"""
dashboard.py
============
Streamlit dashboard: displays portfolio state, statistics, and charts.
Run: streamlit run dashboard.py
WARNING: For research purposes only. No real money involved.
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Agent Analyst – Trading Research Dashboard",
    page_icon="[RESEARCH]",
    layout="wide",
)

RESULTS_DIR = "results"
CACHE_DIR   = "cache"
MODEL_DIR   = "models"

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META",
    "TSLA", "JPM", "V", "BAC", "JNJ", "UNH", "XOM", "WMT", "SPY",
]


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_metrics() -> dict | None:
    path = os.path.join(RESULTS_DIR, "metrics.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data(ttl=300)
def load_equity_curve() -> tuple[list, list] | None:
    path = os.path.join(RESULTS_DIR, "equity_data.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data(ttl=300)
def load_actions_history() -> tuple[list, list, list] | None:
    path = os.path.join(RESULTS_DIR, "actions_data.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def compute_drawdown(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return (peak - equity) / (peak + 1e-9) * 100


# ── Page sections ─────────────────────────────────────────────────────────────

def render_header():
    st.title("Agent Analyst – Trading Research Dashboard")
    st.markdown(
        """
        <div style='background:#fff3cd;padding:12px;border-radius:8px;border-left:4px solid #ffc107;'>
        <b>WARNING:</b> This system is for <b>research and educational purposes only</b>.
        No connection to a real broker. All data is simulated.
        Trading real money requires proper licensing, extensive testing,
        and is the sole responsibility of the user.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")


def render_sidebar():
    with st.sidebar:
        st.header("Settings")
        st.markdown("**Mode:** Paper Trading Stub")
        st.markdown("**Broker:** None (simulated)")
        st.markdown("---")

        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("**Tracked tickers:**")
        for t in TICKERS:
            st.markdown(f"- {t}")

        st.markdown("---")
        st.caption("For research purposes only.")


def render_metrics(metrics: dict):
    st.subheader("Performance Metrics")

    cols = st.columns(5)
    kpis = [
        ("Total Return",   f"{metrics.get('total_return', 0):.1%}"),
        ("Sharpe Ratio",   f"{metrics.get('sharpe', 0):.2f}"),
        ("Max Drawdown",   f"{metrics.get('max_drawdown', 0):.1%}"),
        ("Win Rate",       f"{metrics.get('win_rate', 0):.1%}"),
        ("Profit Factor",  f"{metrics.get('profit_factor', 0):.2f}"),
    ]
    for col, (label, value) in zip(cols, kpis):
        col.metric(label, value)

    cols2 = st.columns(4)
    kpis2 = [
        ("Sortino Ratio",  f"{metrics.get('sortino', 0):.2f}"),
        (f"B&H ({metrics.get('buy_hold_label','SPY')})", f"{metrics.get('buy_hold_return', 0):.1%}"),
        ("Final Equity",   f"${metrics.get('final_equity', 0):,.0f}"),
        ("Total Trades",   str(metrics.get('num_trades', 0))),
    ]
    for col, (label, value) in zip(cols2, kpis2):
        col.metric(label, value)


def render_equity_chart(equity: list, dates: list):
    st.subheader("Equity Curve")
    eq_arr = np.array(equity)
    dd_arr = compute_drawdown(eq_arr)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Portfolio Value ($)", "Drawdown (%)"),
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
    )

    x = dates if dates else list(range(len(eq_arr)))

    fig.add_trace(
        go.Scatter(x=x, y=eq_arr, name="Portfolio",
                   line=dict(color="#2196F3", width=2)),
        row=1, col=1,
    )
    fig.add_hline(y=eq_arr[0],
                  line=dict(dash="dot", color="gray", width=1),
                  row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=x, y=-dd_arr,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#F44336"),
            fillcolor="rgba(244,67,54,0.3)",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis2_title="Date",
        yaxis_title="Value ($)",
        yaxis2_title="Drawdown (%)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)


def render_returns_histogram(equity: list):
    st.subheader("Daily Returns Distribution")
    eq_arr  = np.array(equity)
    returns = np.diff(eq_arr) / (eq_arr[:-1] + 1e-9) * 100

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name="Daily Returns",
        marker_color="#2196F3",
        opacity=0.75,
    ))
    fig.add_vline(
        x=returns.mean(),
        line=dict(color="red", dash="dash"),
        annotation_text=f"Mean = {returns.mean():.2f}%",
    )
    fig.add_vline(x=0, line=dict(color="black", width=1))
    fig.update_layout(
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_actions_heatmap(actions: list, tickers: list, dates: list):
    st.subheader("Actions Heatmap")
    if not actions:
        st.info("No action data available.")
        return

    arr = np.array(actions).T  # (num_stocks, time)
    x   = dates if dates else list(range(arr.shape[1]))

    fig = go.Figure(data=go.Heatmap(
        z=arr,
        x=x,
        y=tickers if tickers else [f"Stock {i}" for i in range(arr.shape[0])],
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Action<br>(-1=Sell, 1=Buy)"),
    ))
    fig.update_layout(
        height=300,
        xaxis_title="Trading Days",
        yaxis_title="Ticker",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_trades_tab():
    """מציג היסטוריית עסקאות מ-trades_history.csv."""
    st.subheader("Trade History")
    csv_path = "trades_history.csv"
    if not os.path.exists(csv_path):
        st.info("No trades recorded yet. trades_history.csv will appear after the first live order.")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        st.info("No trades yet.")
        return

    # סיכום
    buys  = df[df["side"] == "BUY"]
    sells = df[df["side"] == "SELL"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trades", len(df))
    c2.metric("Buys",  len(buys))
    c3.metric("Sells", len(sells))

    # טבלה
    st.dataframe(df.sort_values("time", ascending=False), use_container_width=True, hide_index=True)

    # גרף נפח מסחר
    if "ticker" in df.columns:
        vol = df.groupby("ticker").size().reset_index(name="count").sort_values("count", ascending=False)
        fig = go.Figure(go.Bar(x=vol["ticker"], y=vol["count"],
                               marker_color="#2196F3"))
        fig.update_layout(title="Trades per Ticker", height=300,
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        label="⬇️ Download trades_history.csv",
        data=df.to_csv(index=False),
        file_name="trades_history.csv",
        mime="text/csv",
    )


def render_no_data_placeholder():
    st.info(
        """
        **No simulation results found.**

        Run the simulator first:
        ```
        python main.py --mode simulate
        ```
        Results will appear here automatically once complete.
        """
    )


# ── Live Alpaca data ───────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_live_account() -> dict | None:
    """שולף נתוני חשבון חי מ-Alpaca (cache 60 שניות)."""
    api_key    = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        return None
    try:
        from alpaca.trading.client import TradingClient
        client  = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
        acc     = client.get_account()
        raw_pos = client.get_all_positions()
        positions = {p.symbol: {"qty": float(p.qty), "market_value": float(p.market_value),
                                "unrealized_pl": float(p.unrealized_pl),
                                "current_price": float(p.current_price)}
                     for p in raw_pos}
        return {
            "cash":            float(acc.cash),
            "equity":          float(acc.equity),
            "buying_power":    float(acc.buying_power),
            "portfolio_value": float(acc.portfolio_value),
            "positions":       positions,
        }
    except Exception as exc:
        return {"error": str(exc)}


def render_live_tab():
    st.subheader("Live Account — Alpaca Paper")

    col_refresh, col_auto, _ = st.columns([1, 2, 4])
    with col_refresh:
        if st.button("🔄 Refresh Now"):
            fetch_live_account.clear()
            st.rerun()
    with col_auto:
        auto = st.toggle("Auto-refresh (30s)", value=True)
    if auto:
        import time as _time
        _time.sleep(30)
        fetch_live_account.clear()
        st.rerun()

    data = fetch_live_account()

    if data is None:
        st.warning("ALPACA_API_KEY / ALPACA_SECRET_KEY not found in .env")
        return

    if "error" in data:
        st.error(f"Alpaca API error: {data['error']}")
        return

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity",        f"${data['equity']:,.2f}")
    c2.metric("Cash",          f"${data['cash']:,.2f}")
    c3.metric("Buying Power",  f"${data['buying_power']:,.2f}")
    c4.metric("Portfolio Value", f"${data['portfolio_value']:,.2f}")

    # ── Positions table ───────────────────────────────────────────────────────
    st.markdown("#### Open Positions")
    positions = data["positions"]
    if not positions:
        st.info("No open positions.")
    else:
        rows = []
        for ticker, p in positions.items():
            rows.append({
                "Ticker":          ticker,
                "Shares":          p["qty"],
                "Price":           f"${p['current_price']:,.2f}",
                "Market Value":    f"${p['market_value']:,.2f}",
                "Unrealized P&L":  f"${p['unrealized_pl']:+,.2f}",
            })
        df_pos = pd.DataFrame(rows)
        st.dataframe(df_pos, use_container_width=True, hide_index=True)

        # ── Pie chart positions ───────────────────────────────────────────────
        labels = list(positions.keys())
        values = [positions[t]["market_value"] for t in labels]
        cash_val = data["cash"]
        labels.append("Cash")
        values.append(cash_val)

        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.4,
            marker_colors=["#2196F3","#4CAF50","#FF9800","#9C27B0",
                           "#F44336","#00BCD4","#795548","#607D8B","#E91E63",
                           "#CDDC39","#FF5722","#009688","#3F51B5","#FFC107","#aaa"],
        ))
        fig_pie.update_layout(
            title="Portfolio Allocation",
            height=350,
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.caption("Data from Alpaca Paper API. Use 'Auto-refresh (30s)' toggle or 'Refresh Now' to update.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    render_header()
    render_sidebar()

    tab_live, tab_backtest, tab_actions, tab_trades = st.tabs([
        "🔴 Live Account", "📈 Backtest Results", "🎯 Actions", "📋 Trade History"
    ])

    with tab_live:
        render_live_tab()

    with tab_backtest:
        metrics = load_metrics()
        eq_data = load_equity_curve()

        if metrics:
            render_metrics(metrics)
        else:
            st.warning("No performance metrics found. Run the simulator first.")

        st.markdown("---")

        if eq_data:
            equity, dates = eq_data
            render_equity_chart(equity, dates)
            st.markdown("---")
            render_returns_histogram(equity)
        else:
            render_no_data_placeholder()

        with st.expander("Raw Data"):
            if metrics:
                st.json(metrics)
            if eq_data:
                equity, dates = eq_data
                df_display = pd.DataFrame({
                    "Date":   dates[:100],
                    "Equity": np.array(equity[:100]),
                })
                st.dataframe(df_display)

    with tab_actions:
        act_data = load_actions_history()
        if act_data:
            actions, tickers, dates = act_data
            render_actions_heatmap(actions, tickers, dates)
        else:
            st.info("No action history found. Run the simulator first.")

    with tab_trades:
        render_trades_tab()


if __name__ == "__main__":
    main()
