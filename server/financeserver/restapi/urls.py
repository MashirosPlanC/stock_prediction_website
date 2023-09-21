from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import YFinanceView,MainStockView,QueryStockView,SearchListView,PredictView,GamblingView

router = DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),
    path('yfinance/', YFinanceView.as_view(), name='yfinance'),
    path('stocks/', MainStockView.as_view(), name='stocks'),
    path('query/', QueryStockView.as_view(), name='query'),
    path('searchHist/', SearchListView.as_view(), name='searchHist'),
    path('priDict/', PredictView.as_view(), name='priDict'),
    path('gambling/', GamblingView.as_view(),name='gambling'),
]