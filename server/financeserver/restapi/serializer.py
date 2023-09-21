from rest_framework import serializers

from restapi.models import Books,SearchHist


class BooksSerializer(serializers.ModelSerializer):
    class Meta:
        model = Books
        fields = '__all__'

class SearchHistSerializer(serializers.ModelSerializer):
    class Meta:
        model = SearchHist
        fields = ['searchCode']