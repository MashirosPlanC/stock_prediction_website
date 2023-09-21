import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import yfinance as yf
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

def fetch_stock_prices(stock_symbol, start_date, end_date):
    stock = yf.Ticker(stock_symbol)
    stock_history = stock.history(start=start_date, end=end_date)
    return stock_history["Close"]

# 返回值大概是一个dict，大概长这样：[('AAPL', 0.0), ('GOOGL', 0.01772), ('TSLA', 0.98228), ('AMZN', 0.0)]
# Key 是股票代码，Value 是百分比
def markowitz_portfolio_optimization(stock_data: pd.DataFrame):
    # Calculate expected returns and the covariance matrix of the stock data
    mu = expected_returns.mean_historical_return(stock_data)
    S = risk_models.sample_cov(stock_data)

    # Optimize for the maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    return cleaned_weights

# 调用方法

# stock_list = ["AAPL", "GOOGL", "TSLA", "AMZN"]
# start_date = "2020-01-01"
# end_date = "2021-12-31"
# stock_data = pd.DataFrame()

# for stock in stock_list:
#     stock_data[stock] = fetch_stock_prices(stock, start_date, end_date)
# optimized_weights = markowitz_portfolio_optimization(stock_data)
# print(optimized_weights)

