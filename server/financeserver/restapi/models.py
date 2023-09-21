from django.db import models

# Create your models here.
from django.db import models

class Books(models.Model):
    name = models.CharField(max_length=30)
    author = models.CharField(max_length=30, blank=True, null=True)

class SearchHist(models.Model):
    searchCode = models.CharField(max_length=50)