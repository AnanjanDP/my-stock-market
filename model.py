# model.py
import streamlit as st

# MUST be the very first Streamlit command (before any st.* calls)
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

# -------------------------
# Utility + Caching
# -------------------------
@st.cache_data(ttl=3600)
def fetch_stock_data_cached(ticker: str, start_date: str):
    """
    Cached fetch of OHLC data using yfinance.download.
    Returns a DataFrame or None on failure.
    """
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        # yfinance may raise — return None and show message to user
        return None

@st.cache_resource
def train_model_cached(X: pd.DataFrame, y: pd.Series):
    """
    Train RandomForest once for the same inputs (cached).
    Returns (model, scaler, mean_r2, std_r2)
    """
    # Defensive copies
    Xc = X.fillna(0).copy()
    yc = y.fillna(method='ffill').copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xc)

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = yc.iloc[train_idx], yc.iloc[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        # R2 can be negative; still valid metric
        cv_scores.append(r2_score(y_test, pred))

    mean_r2 = float(np.mean(cv_scores)) if cv_scores else 0.0
    std_r2 = float(np.std(cv_scores)) if cv_scores else 0.0

    return model, scaler, mean_r2, std_r2

# -------------------------
# Analyzer class (keeps your original structure)
# -------------------------
class StockAnalyzer:
    def __init__(self, ticker, start_date="2024-01-01"):
        self.ticker = ticker.upper()
        self.start_date = start_date
        self.stock_data = None

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def fetch_stock_data(self):
        df = fetch_stock_data_cached(self.ticker, self.start_date)
        if df is None or df.empty:
            self.stock_data = None
            return

        # ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        self.stock_data = df.copy()

        # Technical indicators
        self.stock_data['SMA_20'] = self.stock_data['Close'].rolling(window=20).mean()
        self.stock_data['SMA_50'] = self.stock_data['Close'].rolling(window=50).mean()
        self.stock_data['RSI'] = self.calculate_rsi(self.stock_data['Close'], 14)
        self.stock_data['Volatility'] = self.stock_data['Close'].rolling(window=20).std()

        # Fundamental columns — large services changed APIs, provide safe defaults
        # These are placeholders so feature matrix shape remains stable
        self.stock_data['PE_Ratio'] = 0.0
        self.stock_data['Revenue_Growth'] = 0.0
        self.stock_data['Profit_Margin'] = 0.0

    def simulate_sentiment_and_macro(self):
        if self.stock_data is None:
            return

        np.random.seed(42)  # reproducible simulated signals
        dates = self.stock_data.index

        self.stock_data['News_Sentiment'] = pd.Series(np.random.normal(0.2, 0.5, len(dates)), index=dates)
        self.stock_data['GDP_Growth'] = pd.Series(np.random.normal(2.5, 0.5, len(dates)), index=dates)
        self.stock_data['Inflation'] = pd.Series(np.random.normal(2.0, 0.3, len(dates)), index=dates)
        self.stock_data['Interest_Rate'] = pd.Series(np.random.normal(4.5, 0.2, len(dates)), index=dates)

    def prepare_features(self):
        features = [
            'SMA_20', 'SMA_50', 'RSI', 'Volatility',
            'PE_Ratio', 'Revenue_Growth', 'Profit_Margin',
            'News_Sentiment', 'GDP_Growth', 'Inflation', 'Interest_Rate'
        ]
        X = self.stock_data[features].fillna(method='ffill')
        y = self.stock_data['Close']
        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series):
        model, scaler, mean_r2, std_r2 = train_model_cached(X, y)
        return model, scaler, mean_r2, std_r2

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.title("📈 Stock Analysis Dashboard — Updated (2025)")

    with st.sidebar:
        st.header("Configuration")
        ticker = st.text_input("Stock Ticker", value="GOOGL").strip().upper()
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=365)
        )
        analyze_btn = st.button("Analyze Stock")

        st.markdown("---")
        st.caption("Note: fundamentals (PE, margins) use placeholders (yfinance.info is deprecated)."
                   " You can plug an external fundamentals API if you'd like.")

    # Small helpful banner
    st.write("Enter a ticker and click **Analyze Stock**. The app fetches OHLC data, computes indicators, "
             "simulates sentiment/macro features, and trains a Random Forest (cached).")

    if not analyze_btn:
        st.info("Waiting for you to click **Analyze Stock** in the sidebar.")
        return

    # Instantiate analyzer and fetch data
    start_iso = start_date.strftime("%Y-%m-%d")
    analyzer = StockAnalyzer(ticker, start_iso)

    with st.spinner("Fetching stock data..."):
        analyzer.fetch_stock_data()

    if analyzer.stock_data is None or analyzer.stock_data.empty:
        st.error("Failed to fetch data for ticker: " + ticker + ".\n"
                 "Possible causes: invalid ticker, network problem, or Yahoo data not available.")
        return

    # Add simulated sentiment & macro
    analyzer.simulate_sentiment_and_macro()

    # Layout: left wide chart, right model & stats
    col1, col2 = st.columns([2, 1])

    # --- Left: Charts ---
    with col1:
        st.subheader("Technical Analysis")
        try:
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                vertical_spacing=0.05, row_heights=[0.55, 0.25, 0.20]
            )

            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=analyzer.stock_data.index,
                    open=analyzer.stock_data['Open'],
                    high=analyzer.stock_data['High'],
                    low=analyzer.stock_data['Low'],
                    close=analyzer.stock_data['Close'],
                    name="OHLC"
                ),
                row=1, col=1
            )

            # SMAs (only plot if not all-NaN)
            if analyzer.stock_data['SMA_20'].notna().any():
                fig.add_trace(go.Scatter(x=analyzer.stock_data.index, y=analyzer.stock_data['SMA_20'], name="SMA 20"),
                              row=1, col=1)
            if analyzer.stock_data['SMA_50'].notna().any():
                fig.add_trace(go.Scatter(x=analyzer.stock_data.index, y=analyzer.stock_data['SMA_50'], name="SMA 50"),
                              row=1, col=1)

            # RSI + Sentiment on row 2
            fig.add_trace(go.Scatter(x=analyzer.stock_data.index, y=analyzer.stock_data['RSI'], name="RSI"),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=analyzer.stock_data.index, y=analyzer.stock_data['News_Sentiment'], name="Sentiment"),
                          row=2, col=1)

            # Macro indicators on row 3
            fig.add_trace(go.Scatter(x=analyzer.stock_data.index, y=analyzer.stock_data['GDP_Growth'], name="GDP Growth"),
                          row=3, col=1)
            fig.add_trace(go.Scatter(x=analyzer.stock_data.index, y=analyzer.stock_data['Inflation'], name="Inflation"),
                          row=3, col=1)

            fig.update_layout(height=800, showlegend=True, xaxis3_title="Date",
                              yaxis_title="Price", yaxis2_title="Indicators", yaxis3_title="Macro")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error("Error building chart: " + str(e))

    # --- Right: Model & Stats ---
    with col2:
        st.subheader("Model Analysis")

        # Prepare features and train
        X, y = analyzer.prepare_features()

        with st.spinner("Training model (cached for same inputs)..."):
            try:
                model, scaler, cv_mean, cv_std = analyzer.train_model(X, y)
            except Exception as e:
                st.error("Model training failed: " + str(e))
                return

        st.metric(label="Model Performance (R² mean)", value=f"{cv_mean:.3f}", delta=f"±{cv_std:.3f}")

        # Feature importance (safe)
        try:
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)

            fig_importance = go.Figure(go.Bar(x=importance_df['Importance'], y=importance_df['Feature'], orientation='h'))
            fig_importance.update_layout(title="Feature Importance", xaxis_title="Importance Score", height=380)
            st.plotly_chart(fig_importance, use_container_width=True)
        except Exception as e:
            st.warning("Could not compute feature importance: " + str(e))

        # Stock statistics
        st.subheader("Stock Statistics")
        try:
            last_close = float(analyzer.stock_data['Close'].iloc[-1])
            daily_ret = float(analyzer.stock_data['Close'].pct_change().iloc[-1])
            vol20 = float(analyzer.stock_data['Volatility'].iloc[-1])
            rsi_last = float(analyzer.stock_data['RSI'].iloc[-1]) if pd.notna(analyzer.stock_data['RSI'].iloc[-1]) else np.nan
            pe_last = float(analyzer.stock_data['PE_Ratio'].iloc[-1]) if 'PE_Ratio' in analyzer.stock_data.columns else 0.0
        except Exception:
            last_close, daily_ret, vol20, rsi_last, pe_last = (np.nan, np.nan, np.nan, np.nan, 0.0)

        stats = pd.DataFrame({
            'Metric': ['Current Price', 'Daily Return', 'Volatility (20d)', 'RSI', 'PE Ratio'],
            'Value': [
                f"${last_close:.2f}" if not np.isnan(last_close) else "n/a",
                f"{daily_ret:.2%}" if not np.isnan(daily_ret) else "n/a",
                f"{vol20:.2f}" if not np.isnan(vol20) else "n/a",
                f"{rsi_last:.2f}" if not np.isnan(rsi_last) else "n/a",
                f"{pe_last:.2f}"
            ]
        })

        st.dataframe(stats, hide_index=True)

    # Footer / notes
    st.markdown("---")
    st.caption("Built with Streamlit • yfinance (OHLC) • Plotly • scikit-learn.\n"
               "Fundamentals are placeholders because yfinance.info became unreliable. "
               "If you want real fundamentals I can add FinancialModelingPrep / AlphaVantage integration.")

# call main() unconditionally so Streamlit executes the app when imported
main()
