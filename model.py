import streamlit as st
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

class StockAnalyzer:
    def __init__(self, ticker, start_date="2024-01-01"):
        self.ticker = ticker
        self.start_date = start_date
        self.stock_data = None
        
    def fetch_stock_data(self):
        self.stock_data = yf.download(self.ticker, start=self.start_date)
        
        self.stock_data['SMA_20'] = self.stock_data['Close'].rolling(window=20).mean()
        self.stock_data['SMA_50'] = self.stock_data['Close'].rolling(window=50).mean()
        self.stock_data['RSI'] = self.calculate_rsi(self.stock_data['Close'], 14)
        self.stock_data['Volatility'] = self.stock_data['Close'].rolling(window=20).std()
        
        # Fetch fundamental data
        ticker_obj = yf.Ticker(self.ticker)
        try:
            info = ticker_obj.info
            self.stock_data['PE_Ratio'] = pd.Series([info.get('forwardPE', 0)] * len(self.stock_data.index), 
                                                   index=self.stock_data.index)
            self.stock_data['Revenue_Growth'] = pd.Series([info.get('revenueGrowth', 0)] * len(self.stock_data.index), 
                                                         index=self.stock_data.index)
            self.stock_data['Profit_Margin'] = pd.Series([info.get('profitMargins', 0)] * len(self.stock_data.index), 
                                                        index=self.stock_data.index)
        except Exception as e:
            st.error(f"Error fetching fundamental data: {e}")
            self.stock_data['PE_Ratio'] = 0
            self.stock_data['Revenue_Growth'] = 0
            self.stock_data['Profit_Margin'] = 0
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def simulate_sentiment_and_macro(self):
        np.random.seed(42)
        dates = self.stock_data.index
        
        self.stock_data['News_Sentiment'] = pd.Series(
            np.random.normal(0.2, 0.5, len(dates)), 
            index=dates
        )
        self.stock_data['GDP_Growth'] = pd.Series(
            np.random.normal(2.5, 0.5, len(dates)), 
            index=dates
        )
        self.stock_data['Inflation'] = pd.Series(
            np.random.normal(2.0, 0.3, len(dates)), 
            index=dates
        )
        self.stock_data['Interest_Rate'] = pd.Series(
            np.random.normal(4.5, 0.2, len(dates)), 
            index=dates
        )
    
    def prepare_features(self):
        features = [
            'SMA_20', 'SMA_50', 'RSI', 'Volatility',
            'PE_Ratio', 'Revenue_Growth', 'Profit_Margin',
            'News_Sentiment', 'GDP_Growth', 'Inflation', 'Interest_Rate'
        ]
        
        X = self.stock_data[features].fillna(method='ffill')
        y = self.stock_data['Close']
        return X, y
    
    def train_model(self, X, y):
        X = X.fillna(0)
        y = y.fillna(method='ffill')
        
        tscv = TimeSeriesSplit(n_splits=5)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        
        cv_scores = []
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            cv_scores.append(r2_score(y_test, pred))
        
        return model, scaler, np.mean(cv_scores), np.std(cv_scores)

def main():
    st.set_page_config(layout="wide")
    st.title("Stock Analysis Dashboard")
    
    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Stock Ticker", value="GOOGL")
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=365)
    )
    
    if st.sidebar.button("Analyze Stock"):
        analyzer = StockAnalyzer(ticker, start_date.strftime('%Y-%m-%d'))
        
        with st.spinner("Fetching stock data..."):
            analyzer.fetch_stock_data()
            analyzer.simulate_sentiment_and_macro()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Technical Analysis")
            
            fig = make_subplots(
                rows=3, 
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.5, 0.25, 0.25]
            )
            
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
            
            fig.add_trace(
                go.Scatter(
                    x=analyzer.stock_data.index,
                    y=analyzer.stock_data['SMA_20'],
                    name="SMA 20",
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=analyzer.stock_data.index,
                    y=analyzer.stock_data['SMA_50'],
                    name="SMA 50",
                    line=dict(color='orange')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=analyzer.stock_data.index,
                    y=analyzer.stock_data['RSI'],
                    name="RSI",
                    line=dict(color='purple')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=analyzer.stock_data.index,
                    y=analyzer.stock_data['News_Sentiment'],
                    name="Sentiment",
                    line=dict(color='green')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=analyzer.stock_data.index,
                    y=analyzer.stock_data['GDP_Growth'],
                    name="GDP Growth",
                    line=dict(color='blue')
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=analyzer.stock_data.index,
                    y=analyzer.stock_data['Inflation'],
                    name="Inflation",
                    line=dict(color='red')
                ),
                row=3, col=1
            )
            
            fig.update_layout(
                height=800,
                showlegend=True,
                xaxis3_title="Date",
                yaxis_title="Price",
                yaxis2_title="Indicators",
                yaxis3_title="Macro"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Model Analysis")
            
            X, y = analyzer.prepare_features()
            model, scaler, cv_mean, cv_std = analyzer.train_model(X, y)
            
            st.metric(
                label="Model Performance (R²)", 
                value=f"{cv_mean:.3f}",
                delta=f"±{cv_std:.3f}"
            )
            
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig_importance = go.Figure(
                go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h'
                )
            )
            
            fig_importance.update_layout(
                title="Feature Importance",
                xaxis_title="Importance Score",
                height=400
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
            
            st.subheader("Stock Statistics")
            stats = pd.DataFrame({
                'Metric': [
                    'Current Price',
                    'Daily Return',
                    'Volatility (20d)',
                    'RSI',
                    'PE Ratio'
                ],
                'Value': [
                    f"${analyzer.stock_data['Close'][-1]:.2f}",
                    f"{analyzer.stock_data['Close'].pct_change()[-1]:.2%}",
                    f"{analyzer.stock_data['Volatility'][-1]:.2f}",
                    f"{analyzer.stock_data['RSI'][-1]:.2f}",
                    f"{analyzer.stock_data['PE_Ratio'][-1]:.2f}"
                ]
            })
            
            st.dataframe(stats, hide_index=True)

if __name__ == "__main__":
    main()
