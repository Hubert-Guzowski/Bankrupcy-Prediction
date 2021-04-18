from django.urls import path
from . import views

urlpatterns = [
    path('data/<int:year_number>/', views.dataframe_view, name="dataframe view"),
    path('pca/<int:year_number>/', views.pca_view, name="pca view")
]
