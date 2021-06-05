from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.homepage, name="homepage"),
    path('accounts/', include('django.contrib.auth.urls'))
]
