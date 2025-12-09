import streamlit as st

st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

@st.cache_data(show_spinner=False)
def download_stock(ticker: str, start_date: str) -> pd.DataFrame:
    """Download OHLC data using yfinance and return DataFrame (or empty DF)."""
    try:
        df = yf.download(ticker, start=start_date, progress=False, threads=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        return pd.DataFrame()

class StockAnalyzer:
    def __init__(self, ticker: str, start_date: str = "2024-01-01"):
        self.ticker = ticker.upper()
        self.start_date = start_date
        self.stock_data: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
        loss = -delta.clip(upper=0).rolling(window=period, min_periods=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def fetch_stock_data(self):
        df = download_stock(self.ticker, self.start_date)
        if df.empty:
            self.stock_data = df
            return

        df = df.copy()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["RSI"] = self.calculate_rsi(df["Close"], 14)
        df["Volatility"] = df["Close"].rolling(window=20).std()

        df["PE_Ratio"] = 0.0
        df["Revenue_Growth"] = 0.0
        df["Profit_Margin"] = 0.0

        self.stock_data = df

    def simulate_sentiment_and_macro(self, seed: int = 42):
        if self.stock_data.empty:
            return

        np.random.seed(seed)
        dates = self.stock_data.index

        self.stock_data["News_Sentiment"] = pd.Series(
            np.random.normal(0.2, 0.5, len(dates)), index=dates
        )

        self.stock_data["GDP_Growth"] = pd.Series(
            np.random.normal(2.5, 0.5, len(dates)), index=dates
        )
        self.stock_data["Inflation"] = pd.Series(
            np.random.normal(2.0, 0.3, len(dates)), index=dates
        )
        self.stock_data["Interest_Rate"] = pd.Series(
            np.random.normal(4.5, 0.2, len(dates)), index=dates
        )

    def prepare_features(self):
        features = [
            "SMA_20",
            "SMA_50",
            "RSI",
            "Volatility",
            "PE_Ratio",
            "Revenue_Growth",
            "Profit_Margin",
            "News_Sentiment",
            "GDP_Growth",
            "Inflation",
            "Interest_Rate",
        ]

        X = self.stock_data[features].ffill().fillna(0)
        y = self.stock_data["Close"].ffill()
        return X, y

    @st.cache_data(show_spinner=False)
    def train_model(self, X: pd.DataFrame, y: pd.Series):
        """Train RandomForest with TimeSeriesSplit and return model, scaler, and CV statistics.
           This function is cached (depends on X.columns and y.shape implicitly)."""
        if X.empty or y.isna().all():
            return None, None, float("nan"), float("nan")

        X = X.fillna(0)
        y = y.fillna(method="ffill").dropna()
        if y.empty:
            return None, None, float("nan"), float("nan")

        tscv = TimeSeriesSplit(n_splits=5)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

        cv_scores = []
        # If the dataset is too small for 5 splits, reduce splits
        n_splits = min(5, max(2, int(len(X) / 10)))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            try:
                score = r2_score(y_test, pred)
            except Exception:
                score = float("nan")
            cv_scores.append(score)

        cv_mean = float(np.nanmean(cv_scores))
        cv_std = float(np.nanstd(cv_scores))
        return model, scaler, cv_mean, cv_std


def main():
    st.title("📈 Stock Analysis Dashboard (Updated 2025)")
    st.markdown(
        "Interactive technical + simulated sentiment + ML predictions. "
        "If a ticker returns no data, try a different ticker or date range."
    )

    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Stock Ticker", value="GOOGL").strip().upper()
    start_dt = st.sidebar.date_input(
        "Start Date", value=(datetime.now() - timedelta(days=365)).date()
    )
    analyze_btn = st.sidebar.button("Analyze Stock")

    st.sidebar.markdown("Built: 2024 logic — updated for 2025. ✅")

    if not analyze_btn:
        st.info("Enter ticker and click **Analyze Stock**.")
        return

    if ticker == "":
        st.error("Please enter a ticker symbol (e.g., GOOGL, AAPL).")
        return

    analyzer = StockAnalyzer(ticker, start_dt.strftime("%Y-%m-%d"))
    with st.spinner("Fetching stock data..."):
        analyzer.fetch_stock_data()

    if analyzer.stock_data.empty:
        st.error(f"No data found for ticker '{ticker}'. Try another ticker or a different start date.")
        return

    analyzer.simulate_sentiment_and_macro()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"{ticker} — Price & Indicators")
        df = analyzer.stock_data.copy()

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
        )

        # OHLC
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        # SMAs
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20", line=dict(width=1)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50", line=dict(width=1)),
            row=1,
            col=1,
        )

        # RSI + Sentiment
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(width=1)),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["News_Sentiment"], name="Sentiment", line=dict(width=1)),
            row=2,
            col=1,
        )

        # Macro indicators
        fig.add_trace(
            go.Scatter(x=df.index, y=df["GDP_Growth"], name="GDP Growth", line=dict(width=1)),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Inflation"], name="Inflation", line=dict(width=1)),
            row=3,
            col=1,
        )

        fig.update_layout(
            height=800,
            showlegend=True,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Model Analysis")

        X, y = analyzer.prepare_features()

        with st.spinner("Training model (Random Forest) with TimeSeries CV..."):
            model, scaler, cv_mean, cv_std = analyzer.train_model(X, y)

        if model is None:
            st.warning("Not enough data to train the model.")
        else:
            st.metric(label="Model Performance (R² mean)", value=f"{cv_mean:.3f}", delta=f"±{cv_std:.3f}")

            importance_df = pd.DataFrame(
                {"Feature": X.columns, "Importance": model.feature_importances_}
            ).sort_values("Importance", ascending=True)

            fig_imp = go.Figure(
                go.Bar(x=importance_df["Importance"], y=importance_df["Feature"], orientation="h")
            )
            fig_imp.update_layout(title="Feature Importance", height=380, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_imp, use_container_width=True)

        st.subheader("Stock Statistics")
        last_close = float(df["Close"].iloc[-1])
        last_return = df["Close"].pct_change().iloc[-1] if len(df) > 1 else 0.0
        last_vol = float(df["Volatility"].iloc[-1]) if "Volatility" in df.columns else 0.0
        last_rsi = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else 0.0
        last_pe = float(df["PE_Ratio"].iloc[-1]) if "PE_Ratio" in df.columns else 0.0

        stats = pd.DataFrame(
            {
                "Metric": ["Current Price", "Daily Return", "Volatility (20d)", "RSI", "PE Ratio"],
                "Value": [
                    f"${last_close:.2f}",
                    f"{last_return:.2%}",
                    f"{last_vol:.2f}",
                    f"{last_rsi:.2f}",
                    f"{last_pe:.2f}",
                ],
            }
        )
        st.dataframe(stats, hide_index=True, width=320)

    st.markdown("---")
    st.caption("Notes: Fundamentals via yfinance.info were removed because the endpoint is unreliable. "
               "Sentiment & macro values are simulated for demo purposes. For production, plug a real sentiment API or macro data source.")

main()
