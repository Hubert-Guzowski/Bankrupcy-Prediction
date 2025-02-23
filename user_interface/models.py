from django.contrib.auth.models import User
from django.db import models


class Measurement(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField()
    value = models.FloatField()
