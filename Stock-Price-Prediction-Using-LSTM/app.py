from __future__ import annotations

from datetime import date

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from backend.predictor import load_prediction_model, predict_stock

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_prediction_model()


@st.cache_data(show_spinner=False)
def get_prediction_data(ticker: str, n_forecast_days: int = 30):
    return predict_stock(ticker, get_model(), n_forecast_days=n_forecast_days)


def plot_close_and_averages(history):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(history["Date"], history["Close"], label="Close Price", linewidth=2, color="#0f766e")
    ax.plot(history["Date"], history["MA100"], label="100-Day MA", linewidth=1.6, color="#ea580c")
    ax.plot(history["Date"], history["MA200"], label="200-Day MA", linewidth=1.6, color="#7c2d12")
    ax.set_title("Closing Price with Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.tick_params(axis="x", colors="#000000")
    ax.tick_params(axis="y", colors="#000000")
    ax.xaxis.label.set_color("#000000")
    ax.yaxis.label.set_color("#000000")
    ax.title.set_color("#000000")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.patch.set_alpha(0)
    ax.set_facecolor("#fffaf4")
    fig.autofmt_xdate()
    return fig


def build_candlestick_chart(history, ticker: str):
    recent = history.tail(120).copy()
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=recent["Date"],
                open=recent["Open"],
                high=recent["High"],
                low=recent["Low"],
                close=recent["Close"],
                increasing_line_color="#0f766e",
                decreasing_line_color="#b91c1c",
                name="Price",
            ),
            go.Scatter(
                x=recent["Date"],
                y=recent["MA100"],
                mode="lines",
                line={"color": "#ea580c", "width": 2},
                name="MA100",
            ),
            go.Scatter(
                x=recent["Date"],
                y=recent["MA200"],
                mode="lines",
                line={"color": "#7c2d12", "width": 2},
                name="MA200",
            ),
        ]
    )
    fig.update_layout(
        title=f"{ticker} Candlestick View",
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": 1.08, "x": 0},
        font={"color": "#000000"},
        height=430,
    )
    fig.update_layout(
        title_font={"color": "#000000"},
        legend_font={"color": "#000000"},
    )
    fig.update_xaxes(showgrid=False, tickfont={"color": "#000000"}, title_font={"color": "#000000"})
    fig.update_yaxes(gridcolor="rgba(31, 26, 23, 0.08)", tickfont={"color": "#000000"}, title_font={"color": "#000000"})
    return fig


def build_future_forecast_chart(result, n_days: int):
    import pandas as pd
    last_actual_date = result.history["Date"].iloc[-1]
    history_tail = result.history[["Date", "Close"]].tail(60).copy()

    future_df = pd.DataFrame({
        "Date": pd.to_datetime(result.forecast_dates),
        "Forecast": result.forecast_prices[:n_days],
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_tail["Date"],
        y=history_tail["Close"],
        mode="lines",
        name="Historical Close",
        line={"color": "#1d4ed8", "width": 2.5},
    ))
    fig.add_trace(go.Scatter(
        x=future_df["Date"],
        y=future_df["Forecast"],
        mode="lines+markers",
        name="Forecasted Close",
        line={"color": "#c2410c", "width": 2.5, "dash": "dot"},
        marker={"size": 5, "color": "#c2410c"},
    ))
    # shaded forecast zone
    fig.add_vrect(
        x0=str(future_df["Date"].iloc[0].date()),
        x1=str(future_df["Date"].iloc[-1].date()),
        fillcolor="rgba(194,65,12,0.07)",
        layer="below",
        line_width=0,
        annotation_text="Forecast Zone",
        annotation_position="top left",
        annotation_font_color="#c2410c",
    )
    fig.update_layout(
        title=f"{result.ticker} — {n_days}-Day Future Price Forecast",
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        font={"color": "#000000"},
        height=450,
    )
    fig.update_xaxes(showgrid=False, tickfont={"color": "#000000"}, title_font={"color": "#000000"})
    fig.update_yaxes(gridcolor="rgba(31,26,23,0.08)", tickfont={"color": "#000000"}, title_font={"color": "#000000"})
    return fig, future_df


def build_prediction_chart(test_data, ticker: str):
    recent = test_data.tail(120).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent["Date"],
            y=recent["Actual Close"],
            mode="lines",
            line={"color": "#1d4ed8", "width": 3},
            name="Actual Close",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recent["Date"],
            y=recent["Predicted Close"],
            mode="lines",
            line={"color": "#c2410c", "width": 3, "dash": "dash"},
            name="Predicted Close",
        )
    )
    fig.update_layout(
        title=f"{ticker} Actual vs Predicted Close",
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        font={"color": "#D61F1F"},
        height=430,
    )
    fig.update_xaxes(showgrid=False, tickfont={"color": "#000000"}, title_font={"color": "#000000"})
    fig.update_yaxes(gridcolor="rgba(31, 26, 23, 0.08)", tickfont={"color": "#000000"}, title_font={"color": "#000000"})
    return fig


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4efe7;
            --panel: rgba(255, 250, 244, 0.82);
            --panel-strong: rgba(255, 250, 244, 0.95);
            --ink: #1f1a17;
            --muted: #6b6259;
            --accent: #0e7490;
            --accent-2: #c2410c;
            --good: #0f766e;
            --bad: #b91c1c;
            --line: rgba(31, 26, 23, 0.08);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(14, 116, 144, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(194, 65, 12, 0.14), transparent 26%),
                linear-gradient(180deg, #f9f4ec 0%, var(--bg) 100%);
            color: var(--ink);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1240px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(255,250,244,0.95), rgba(255,244,231,0.92));
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 1.9rem 1.7rem;
            box-shadow: 0 24px 70px rgba(31, 26, 23, 0.08);
            margin-bottom: 1.2rem;
            animation: fadeUp 0.6s ease-out;
        }
        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.72rem;
            color: var(--accent);
            font-weight: 700;
            margin-bottom: 0.6rem;
        }
        .hero-title {
            font-size: 2.55rem;
            line-height: 1.02;
            font-weight: 800;
            margin: 0;
            color: var(--ink);
        }
        .hero-copy {
            margin-top: 0.9rem;
            max-width: 760px;
            color: var(--muted);
            font-size: 1rem;
        }
        .insight-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1.2rem;
        }
        .insight-card, .ticker-banner, .forecast-panel, .stat-chip {
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(31, 26, 23, 0.07);
            box-shadow: 0 12px 35px rgba(31, 26, 23, 0.05);
        }
        .insight-card {
            border-radius: 18px;
            padding: 1rem;
        }
        .insight-label, .mini-label {
            color: var(--muted);
            font-size: 0.84rem;
        }
        .insight-value {
            margin-top: 0.35rem;
            font-size: 1.12rem;
            font-weight: 700;
            color: var(--ink);
        }
        .section-label, .result-label {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 700;
        }
        .section-label {
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            color: var(--ink);
            font-size: 0.88rem;
        }
        .snapshot-card {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-left: 6px solid var(--accent);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            box-shadow: 0 10px 25px rgba(31, 26, 23, 0.05);
            margin-bottom: 1rem;
        }
        .snapshot-title {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.74rem;
            color: var(--accent);
            font-weight: 800;
            margin-bottom: 0.35rem;
        }
        .snapshot-text {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 600;
        }
        .result-label {
            margin-top: 1.1rem;
            margin-bottom: 0.65rem;
            font-size: 0.78rem;
            color: var(--accent-2);
        }
        .ticker-banner {
            border-radius: 24px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            animation: fadeUp 0.75s ease-out;
        }
        .ticker-grid {
            display: grid;
            grid-template-columns: 72px 1.2fr 1fr 1fr;
            gap: 1rem;
            align-items: center;
        }
        .ticker-badge {
            width: 72px;
            height: 72px;
            border-radius: 20px;
            background: linear-gradient(135deg, #0e7490, #1d4ed8);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4rem;
            font-weight: 800;
        }
        .ticker-name {
            font-size: 1.4rem;
            font-weight: 800;
            color: var(--ink);
            margin-bottom: 0.2rem;
        }
        .ticker-sub {
            color: var(--muted);
            font-size: 0.92rem;
        }
        .pulse {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 999px;
            background: var(--good);
            margin-right: 0.45rem;
            box-shadow: 0 0 0 rgba(15, 118, 110, 0.5);
            animation: pulse 1.8s infinite;
        }
        .stat-chip {
            border-radius: 18px;
            padding: 0.9rem 1rem;
        }
        .stat-value {
            margin-top: 0.25rem;
            font-size: 1.12rem;
            font-weight: 700;
            color: var(--ink);
        }
        .stMetric {
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.8rem;
            box-shadow: 0 10px 30px rgba(31, 26, 23, 0.05);
        }
        div[data-testid="stMetricLabel"] {
            color: blue;
            font-weight: 700;
        }
        div[data-testid="stMetricValue"] {
            color: #1d4ed8;
        }
        div[data-testid="stMetricDelta"] {
            color: #1d4ed8;
        }
        .chart-shell {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 0.7rem 0.7rem 0.2rem 0.7rem;
            box-shadow: 0 12px 35px rgba(31, 26, 23, 0.05);
            animation: fadeUp 0.8s ease-out;
        }
        .forecast-panel {
            border-radius: 24px;
            padding: 1.1rem;
            height: 100%;
            animation: fadeUp 0.85s ease-out;
        }
        .signal-pill {
            display: inline-block;
            padding: 0.32rem 0.75rem;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 700;
            margin: 0.4rem 0 0.75rem 0;
            color: white;
            background: linear-gradient(135deg, #0f766e, #14b8a6);
        }
        .signal-pill.bearish {
            background: linear-gradient(135deg, #b91c1c, #ef4444);
        }
        .summary-list {
            display: grid;
            gap: 0.75rem;
            margin-top: 0.7rem;

        }
        .summary-row {
            border-top: 1px solid var(--line);
            padding-top: 0.75rem;
        }
        .small-note {
            color: var(--muted);
            font-size: 0.86rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: rgba(255,255,0,0.55);
            padding: 0.5rem 1rem;
            color: var(--ink);
            border: 1px solid rgba(31, 26, 23, 0.07);
        }
        .st-emotion-cache-1p09rwb {
            color: #000000 !important;
        }
        @keyframes fadeUp {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        @keyframes pulse {
            0% {
                box-shadow: 0 0 0 0 rgba(15, 118, 110, 0.45);
            }
            70% {
                box-shadow: 0 0 0 14px rgba(15, 118, 110, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(15, 118, 110, 0);
            }
        }
        @media (max-width: 980px) {
            .hero-title {
                font-size: 1.85rem;
            }
            .insight-strip, .ticker-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">LSTM Forecast Studio</div>
            <h1 class="hero-title">Stock Price Prediction Dashboard</h1>
            <div class="hero-copy">
                This dashboard presents real-time stock data from Yahoo Finance, combined with LSTM-based forecasting and clear visual insights into predicted trends up to {date.today().isoformat()}.
            </div>
            <div class="insight-strip">
                <div class="insight-card">
                    <div class="insight-label">Model Input Window</div>
                    <div class="insight-value">100 trading days</div>
                </div>
                <div class="insight-card">
                    <div class="insight-label">Prediction Source</div>
                    <div class="insight-value">Saved Keras LSTM model</div>
                </div>
                <div class="insight-card">
                    <div class="insight-label">Market Feed</div>
                    <div class="insight-value">Yahoo Finance</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ticker_banner(result, delta_percent: float):
    latest_date = result.history["Date"].iloc[-1].date().isoformat()
    volume = int(result.history["Volume"].iloc[-1]) if "Volume" in result.history.columns else 0
    st.markdown(
        f"""
        <div class="ticker-banner">
            <div class="ticker-grid">
                <div class="ticker-badge">{result.ticker[:2]}</div>
                <div>
                    <div class="ticker-name">{result.ticker}</div>
                    <div class="ticker-sub"><span class="pulse"></span>Latest market session: {latest_date}</div>
                </div>
                <div class="stat-chip">
                    <div class="mini-label">Daily Volume</div>
                    <div class="stat-value">{volume:,}</div>
                </div>
                <div class="stat-chip">
                    <div class="mini-label">Projected Move</div>
                    <div class="stat-value">{delta_percent:+.2f}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_styles()
render_hero()

with st.sidebar:
    st.markdown("## Prediction Settings")
    st.caption("Choose a stock ticker and refresh the dashboard.")
    quick_pick = st.selectbox("Popular tickers", [
    # 🇺🇸 US Stocks 
    "AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "META", "NFLX",
    "NVDA", "AMD", "INTC", "PYPL", "ADBE", "ORCL", "CSCO",

    # 🇮🇳 Indian Stocks 
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "SBIN.NS", "LT.NS", "ITC.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "BAJFINANCE.NS",
    "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "WIPRO.NS", "ULTRACEMCO.NS", "TITAN.NS","VEDL.NS"
], index=0)
    ticker = st.text_input("Ticker symbol", value=quick_pick, help="Examples: AAPL, MSFT, TSLA, GOOG, NFLX")
    n_forecast_days = st.slider(
        "Forecast horizon (trading days)",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
        help="Number of future trading days to predict after today's last known close.",
    )
    run_prediction = st.button("Run Prediction", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("### What You See")
    st.markdown(
        """
        <div class="small-note">
            After prediction, the dashboard labels each panel clearly:
            market structure, model comparison, forecast summary, and recent rows used for comparison.
        </div>
        """,
        unsafe_allow_html=True,
    )

if run_prediction or ticker:
    cleaned_ticker = ticker.strip().upper()

    if not cleaned_ticker:
        st.warning("Enter a valid ticker symbol to continue.")
    else:
        try:
            with st.spinner(f"Fetching data and running the LSTM model for {cleaned_ticker}..."):
                result = get_prediction_data(cleaned_ticker, n_forecast_days)

            change = result.next_close_prediction - result.latest_close
            delta_percent = (change / result.latest_close) * 100 if result.latest_close else 0.0
            high_price = float(result.history["High"].tail(30).max())
            low_price = float(result.history["Low"].tail(30).min())
            avg_close = float(result.history["Close"].tail(30).mean())

            render_ticker_banner(result, delta_percent)

            st.markdown(
                f"""
                <div class="snapshot-card">
                    <div class="snapshot-title">Prediction Snapshot</div>
                    <div class="snapshot-text">
                        Snapshot for <strong>{result.ticker}</strong> using <strong>{result.model_input_rows}</strong> rows of historical data.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metric_1, metric_2, metric_3, metric_4 = st.columns(4)
            metric_1.metric("Latest Close", f"${result.latest_close:,.2f}")
            metric_2.metric("Predicted Next Close", f"${result.next_close_prediction:,.2f}")
            metric_3.metric("Expected Change", f"{change:,.2f}", f"{delta_percent:.2f}%")
            metric_4.metric("30-Day Avg Close", f"${avg_close:,.2f}")

            overview_tab, prediction_tab, forecast_tab, data_tab = st.tabs(
                ["Market Overview", "Prediction View", "📅 Future Forecast", "Recent Data"]
            )

            with overview_tab:
                st.markdown('<div class="result-label">Candlestick Trend and Moving Averages</div>', unsafe_allow_html=True)
                st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
                st.plotly_chart(build_candlestick_chart(result.history, result.ticker), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                info_1, info_2, info_3 = st.columns(3)
                info_1.info(f"30-day high: ${high_price:,.2f}")
                info_2.info(f"30-day low: ${low_price:,.2f}")
                info_3.info("Candles show open, high, low, and close for each trading day.")

                st.markdown('<div class="result-label">Full History Close Trend</div>', unsafe_allow_html=True)
                st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
                st.pyplot(plot_close_and_averages(result.history), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with prediction_tab:
                left_col, right_col = st.columns([2.1, 1])
                with left_col:
                    st.markdown('<div class="result-label">Actual Price vs Model Prediction</div>', unsafe_allow_html=True)
                    st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
                    st.plotly_chart(build_prediction_chart(result.test_data, result.ticker), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with right_col:
                    trend_text = "Bullish bias" if change >= 0 else "Bearish bias"
                    confidence_note = (
                        "The next predicted close is above the latest market close."
                        if change >= 0
                        else "The next predicted close is below the latest market close."
                    )
                    pill_class = "signal-pill" if change >= 0 else "signal-pill bearish"
                    st.markdown(
                        f"""
                        <div class="forecast-panel">
                            <div class="result-label">Forecast Summary</div>
                            <div class="{pill_class}">{trend_text}</div>
                            <div class="summary-list">
                                <div class="summary-row">
                                    <div class="mini-label">Ticker</div>
                                    <div class="stat-value">{result.ticker}</div>
                                </div>
                                <div class="summary-row">
                                    <div class="mini-label">Model Signal</div>
                                    <div class="stat-value">{change:+.2f} USD</div>
                                </div>
                                <div class="summary-row">
                                    <div class="mini-label">Interpretation</div>
                                    <div class="small-note">{confidence_note}</div>
                                </div>
                                <div class="summary-row">
                                    <div class="mini-label">Reminder</div>
                                    <div class="small-note">
                                        This is a directional estimate from the trained notebook model, not financial advice.
                                    </div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with forecast_tab:
                st.markdown(
                    f'<div class="result-label">LSTM {n_forecast_days}-Day Future Price Forecast</div>',
                    unsafe_allow_html=True,
                )
                fc_chart, future_df = build_future_forecast_chart(result, n_forecast_days)
                st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
                st.plotly_chart(fc_chart, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                fc_col1, fc_col2, fc_col3 = st.columns(3)
                fc_col1.metric("Forecast Start", str(future_df["Date"].iloc[0].date()))
                fc_col2.metric("Forecast End", str(future_df["Date"].iloc[-1].date()))
                fc_end_price = future_df["Forecast"].iloc[-1]
                fc_change_pct = ((fc_end_price - result.latest_close) / result.latest_close) * 100
                fc_col3.metric(
                    f"Price on Day {n_forecast_days}",
                    f"${fc_end_price:,.2f}",
                    f"{fc_change_pct:+.2f}% vs today",
                )

                st.markdown('<div class="result-label">Daily Forecast Table</div>', unsafe_allow_html=True)
                display_df = future_df.copy()
                display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
                display_df["Forecast Price (USD)"] = display_df["Forecast"].map(lambda x: f"${x:,.2f}")
                display_df["Change vs Today"] = display_df["Forecast"].map(
                    lambda x: f"{((x - result.latest_close)/result.latest_close)*100:+.2f}%"
                )
                st.dataframe(
                    display_df[["Date", "Forecast Price (USD)", "Change vs Today"]].reset_index(drop=True),
                    use_container_width=True,
                    height=320,
                )

            with data_tab:
                st.markdown('<div class="result-label">Latest Rows Used for Comparison</div>', unsafe_allow_html=True)
                st.dataframe(
                    result.test_data[["Date", "Actual Close", "Predicted Close"]]
                    .tail(20)
                    .reset_index(drop=True),
                    use_container_width=True,
                )
        except Exception as exc:
            st.error(str(exc))
            st.stop()
