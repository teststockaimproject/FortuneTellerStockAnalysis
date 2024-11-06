import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Define available stock tickers for the dropdown
PREDEFINED_TICKERS = {
    "Coca-Cola (KO)": "KO",
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "Amazon (AMZN)": "AMZN",
}

# Define available technical analysis techniques
ANALYSIS_TECHNIQUES = ["Golden/Death Cross", "RSI", "MACD"]

# Load stock data
def load_stock_data(ticker: str, start: str, end: str):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data

# Golden Cross/Death Cross Signals
def add_golden_death_cross_signals(stock_data: pd.DataFrame):
    stock_data['30-Day MA'] = stock_data['Close'].rolling(window=30).mean()
    stock_data['90-Day MA'] = stock_data['Close'].rolling(window=90).mean()
    stock_data['Signal'] = 0
    stock_data.loc[stock_data['30-Day MA'] > stock_data['90-Day MA'], 'Signal'] = 1  # Golden Cross (Buy)
    stock_data.loc[stock_data['30-Day MA'] < stock_data['90-Day MA'], 'Signal'] = -1  # Death Cross (Sell)
    return stock_data.dropna()

# RSI Calculation
def add_rsi_signals(stock_data: pd.DataFrame, window: int = 14):
    delta = stock_data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))

    # Set RSI signals
    stock_data['Signal'] = 0
    stock_data.loc[stock_data['RSI'] > 70, 'Signal'] = -1  # Overbought (Sell)
    stock_data.loc[stock_data['RSI'] < 30, 'Signal'] = 1   # Oversold (Buy)
    return stock_data.dropna()

# MACD Calculation
def add_macd_signals(stock_data: pd.DataFrame):
    stock_data['12-Day EMA'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
    stock_data['26-Day EMA'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = stock_data['12-Day EMA'] - stock_data['26-Day EMA']
    stock_data['Signal Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

    # Set MACD signals
    stock_data['Signal'] = 0
    stock_data.loc[stock_data['MACD'] > stock_data['Signal Line'], 'Signal'] = 1  # Bullish crossover (Buy)
    stock_data.loc[stock_data['MACD'] < stock_data['Signal Line'], 'Signal'] = -1 # Bearish crossover (Sell)
    return stock_data.dropna()

# Train the model
def train_signal_model(stock_data: pd.DataFrame):
    """
    Trains a logistic regression model to predict signals based on the selected technical analysis.

    Parameters:
    - stock_data (pd.DataFrame): Data with 'Close' prices and 'Signal'.

    Returns:
    - model (LogisticRegression or None): Trained logistic regression model if successful, else None.
    - X_test (pd.DataFrame): Test features.
    - y_test (pd.Series): Test target values.
    - y_pred (np.array or None): Predictions for the test set if training was successful, else None.
    """
    X = stock_data[['Close']]
    y = stock_data['Signal']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Check if we have at least two classes in the training data
    if len(y_train.unique()) < 2:
        st.warning("Insufficient class variety for training: only one signal type detected.")
        return None, X_test, y_test, None

    # Initialize and train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred

# Plot stock signals
def plot_signals(stock_data: pd.DataFrame, analysis_type: str):
    # Plot Close Price
    fig, ax = plt.subplots(3, 1, figsize=(14, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
    ax[0].plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
    ax[0].set_title('Stock Close Price')
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Price (USD)')
    ax[0].legend()

    if analysis_type == "Golden/Death Cross":
        ax[0].plot(stock_data.index, stock_data['30-Day MA'], label='30-Day MA', color='orange', linestyle='--')
        ax[0].plot(stock_data.index, stock_data['90-Day MA'], label='90-Day MA', color='green', linestyle='--')
        ax[0].legend()
    elif analysis_type == "RSI":
        ax[1].plot(stock_data.index, stock_data['RSI'], label='RSI', color='purple')
        ax[1].set_title('Relative Strength Index (RSI)')
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('RSI Value')
        ax[1].axhline(70, color='red', linestyle='--', linewidth=1)
        ax[1].axhline(30, color='green', linestyle='--', linewidth=1)
        ax[1].legend()
    elif analysis_type == "MACD":
        ax[2].plot(stock_data.index, stock_data['MACD'], label='MACD', color='orange')
        ax[2].plot(stock_data.index, stock_data['Signal Line'], label='Signal Line', color='green')
        ax[2].set_title('MACD Indicator')
        ax[2].set_xlabel('Date')
        ax[2].set_ylabel('MACD Value')
        ax[2].legend()

    # Plot buy/sell signals on the closing price graph
    buy_signals = stock_data[stock_data['Signal'] == 1]
    sell_signals = stock_data[stock_data['Signal'] == -1]
    ax[0].scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', alpha=1)
    ax[0].scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', alpha=1)
    ax[0].legend()

    plt.tight_layout()
    st.pyplot(fig)  # Display the plot in Streamlit

# Main function for the Streamlit app
def main():
    st.title("Stock Analysis with Technical Indicators")
    st.write("Choose a stock, date range, and analysis technique to view technical indicators and signals.")

    # Dropdown for selecting a stock ticker
    ticker_name = st.selectbox("Select a Stock:", list(PREDEFINED_TICKERS.keys()))
    ticker = PREDEFINED_TICKERS[ticker_name]

    # Input fields for date range
    start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))

    # Dropdown for selecting analysis technique
    analysis_type = st.selectbox("Select Technical Analysis Technique:", ANALYSIS_TECHNIQUES)

    if start_date >= end_date:
        st.error("End date must be after start date.")
    else:
        # Load and process stock data
        stock_data = load_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        # Apply the selected technical analysis technique
        if analysis_type == "Golden/Death Cross":
            stock_data_with_signals = add_golden_death_cross_signals(stock_data)
        elif analysis_type == "RSI":
            stock_data_with_signals = add_rsi_signals(stock_data)
        elif analysis_type == "MACD":
            stock_data_with_signals = add_macd_signals(stock_data)

        # Train the model and get predictions
        model, X_test, y_test, y_pred = train_signal_model(stock_data_with_signals)

        # Display classification report and confusion matrix
        st.subheader("Classification Report")
        if y_pred is not None:
            st.text(classification_report(y_test, y_pred))
            st.subheader("Confusion Matrix")
            st.text(confusion_matrix(y_test, y_pred))

        # Plot the stock data with signals
        plot_signals(stock_data_with_signals, analysis_type)

# Run the app
if __name__ == "__main__":
    main()
