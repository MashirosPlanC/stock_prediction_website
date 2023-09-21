from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import generics
import yfinance as yf
from yahooquery import Ticker
from .serializer import SearchHistSerializer
from .models import SearchHist
from datetime import date, datetime, timedelta
from predict import predict_stock_prices_lstm
from portfolio import fetch_stock_prices,markowitz_portfolio_optimization
import pandas as pd
import sys

from datetime import datetime


# import yahoo_fin.stock_info as si
# from yahoo_fin.stock_info import get_analysts_info
# from yahoo_fin import options

class YFinanceView(APIView):
    def get(self, request):
        code = request.GET.get('code')

        print("code", code)
        search = SearchHist(searchCode=code)
        search.save()
        data = {}
        stock = yf.Ticker(code)
        hist = stock.history(period="1mo")
        lastPrice = stock.history()['Close'].iloc[-1]
        data["info"] = stock.info
        data["info"]["lastPrice"] = lastPrice
        data["hist"] = hist
        data["hist"]["Date"] = hist.index
        # data["regularMarketPrice"] = stock.info['regularMarketPrice']
        # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        return Response(data)


class MainStockView(APIView):
    def get(self, request):
        # mainStockCode = ["^GSPC","^DJI","^IXIC","^RUT","GC=F"]
        # mainStockCode = request.query_params.getlist('codes')
        mainStockCode = request.query_params.getlist('codes[]')
        print("mainStockCode ", request.query_params)
        arrInfo = []
        for code in mainStockCode:
            print("code", code)
            stock = yf.Ticker(code)
            lastPrice = stock.history()['Close'].iloc[-1]
            data = {}
            data["info"] = stock.info
            data["info"]["lastPrice"] = lastPrice
            arrInfo.append(data)

        # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        # print(arrInfo)
        return Response(arrInfo)


class QueryStockView(APIView):
    def get(self, request):
        mainStockCode = request.query_params.getlist('codes[]')
        # print("mainStockCode >>>  ",len(mainStockCode))
        if len(mainStockCode) == 0:
            mainStockCode = ["^GSPC", "^DJI", "^IXIC", "^RUT", "GC=F"]

        stocks = Ticker(mainStockCode)
        data = {}
        # data["summary_detail"] = stocks.summary_detail
        data["price"] = stocks.price
        return Response(data)


class SearchListView(APIView):
    def get(self, request):
        queryset = SearchHist.objects.all().order_by('-id')[:10]
        serializer = SearchHistSerializer(queryset, many=True)
        print("hist >> ", serializer.data)
        return Response(serializer.data)

class GamblingView(APIView):
    def get(self, request):
        # stock_list = ["AAPL", "GOOGL", "TSLA", "AMZN"]
        # start_date = "2020-01-01"
        # end_date = "2021-12-31"

        strCodes = request.GET.get('codes')
        print("str codes >> ",strCodes)
        stock_list = strCodes.split(",")
        stock_data = pd.DataFrame()

        today = date.today()
        year = today.year
        month = today.month
        day = today.day

        end_date = str(year) + "-" + str(month) + "-" + str(day)
        delta = timedelta(days=365)
        future_date = today - delta
        future_year = future_date.year
        future_month = future_date.month
        future_day = future_date.day
        start_date = str(future_year) + "-" + str(future_month) + "-" + str(future_day)

        for stock in stock_list:
            stock_data[stock] = fetch_stock_prices(stock, start_date, end_date)
        optimized_weights = markowitz_portfolio_optimization(stock_data)
        print(optimized_weights)

        return Response(optimized_weights)
class PredictView(APIView):
    def get(self, request):
        code = request.GET.get('code')
        # prices = request.query_params.getlist('prices[]')
        # prices_series = pd.Series(prices)
        # print("prices", prices)
        stock_symbol = code
        today = date.today()
        year = today.year
        month = today.month
        day = today.day

        end_date = str(year)+"-"+str(month)+"-"+str(day)
        delta = timedelta(days=365)
        future_date = today - delta
        future_year = future_date.year
        future_month = future_date.month
        future_day = future_date.day
        start_date = str(future_year)+"-"+str(future_month)+"-"+str(future_day)

        stock_prices = fetch_stock_prices(stock_symbol, start_date, end_date)
        print("dates >>>", stock_prices)
        days_to_predict = 30
        result = predict_stock_prices_lstm(stock_prices, days_to_predict)
        """
        result = [('2021-12-31', 137.26192),
                  ('2022-01-01', 137.06328),
                  ('2022-01-02', 137.1638),
                  ('2022-01-03', 137.4514),
                  ('2022-01-04', 137.7359),
                  ('2022-01-05', 137.79018),
                  ('2022-01-06', 137.84067),
                  ('2022-01-07', 138.18294),
                  ('2022-01-08', 138.6354),
                  ('2022-01-09', 139.23161),
                  ('2022-01-10', 139.8793),
                  ('2022-01-11', 140.36845),
                  ('2022-01-12', 140.75835),
                  ('2022-01-13', 140.68729),
                  ('2022-01-14', 140.41298),
                  ('2022-01-15', 140.21083),
                  ('2022-01-16', 140.63971),
                  ('2022-01-17', 141.26213),
                  ('2022-01-18', 142.13193),
                  ('2022-01-19', 142.64354),
                  ('2022-01-20', 143.14136),
                  ('2022-01-21', 143.6643),
                  ('2022-01-22', 144.28568),
                  ('2022-01-23', 144.9365),
                  ('2022-01-24', 145.55608),
                  ('2022-01-25', 146.09471),
                  ('2022-01-26', 146.2908),
                  ('2022-01-27', 146.31602),
                  ('2022-01-28', 146.49344),
                  ('2022-01-29', 146.67998)]
        """
        return Response(result)
