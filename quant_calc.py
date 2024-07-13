import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def get_valid_input(prompt, input_type=str):
    while True:
        try:
            user_input = input_type(input(prompt))
            return user_input
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__}.")

def get_stock_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data[ticker] = stock.history(start=start_date, end=end_date)['Close']
    return pd.DataFrame(data)

def calculate_returns(prices):
    return prices.pct_change().dropna()

def calculate_sharpe_ratio(returns, risk_free_rate, trading_days=252):
    excess_returns = returns - risk_free_rate / trading_days
    sharpe_ratio = np.sqrt(trading_days) * excess_returns.mean() / excess_returns.std()
    return sharpe_ratio

def calculate_sortino_ratio(returns, risk_free_rate, trading_days=252):
    downside_returns = returns[returns < 0]
    excess_returns = returns - risk_free_rate / trading_days
    sortino_ratio = np.sqrt(trading_days) * excess_returns.mean() / downside_returns.std()
    return sortino_ratio

def calculate_max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

def save_to_csv(data, filename):
    df = pd.DataFrame(data).T
    df.to_csv(filename, index=True)

def main():
    # User inputs
    tickers = input("Enter stock ticker symbols separated by commas (e.g., AAPL, MSFT, GOOGL): ").split(',')
    tickers = [ticker.strip().upper() for ticker in tickers]  # Clean and capitalize tickers
    start_date = get_valid_input("Enter start date (YYYY-MM-DD): ", str)
    end_date = get_valid_input("Enter end date (YYYY-MM-DD): ", str)
    risk_free_rate = get_valid_input("Enter annual risk-free rate (e.g., 0.02 for 2%): ", float)

    # Fetch stock data
    stock_prices = get_stock_data(tickers, start_date, end_date)
    
    # Calculate returns and financial metrics
    returns = calculate_returns(stock_prices)
    metrics = {}
    for ticker in tickers:
        sharpe_ratio = calculate_sharpe_ratio(returns[ticker], risk_free_rate)
        sortino_ratio = calculate_sortino_ratio(returns[ticker], risk_free_rate)
        max_drawdown = calculate_max_drawdown(returns[ticker])
        metrics[ticker] = {
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown
        }
    
    # Print results
    for ticker, metric in metrics.items():
        print(f"\nMetrics for {ticker}:")
        for key, value in metric.items():
            print(f"{key}: {value:.4f}")
    
    # Save results to CSV
    save_to_csv(metrics, 'financial_metrics.csv')
    print("\nMetrics saved to financial_metrics.csv")
    
    # Plot cumulative returns
    returns.cumsum().plot(figsize=(10, 6))
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend(tickers)
    plt.show()

if __name__ == "__main__":
    main()
