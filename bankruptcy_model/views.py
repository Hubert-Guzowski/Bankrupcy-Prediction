from django.http import HttpResponse
from django.shortcuts import render

from bankruptcy_model.data_visualizations.pca import prepare_data_for_pca
from bankruptcy_model.utils import data_loading


def dataframe_view(request, year_number):
    dataframe = data_loading.load_dataset_by_year(year_number)
    return HttpResponse(dataframe.to_html())


def pca_view(request, year_number):
    results = prepare_data_for_pca(year_number)
    return render(request, 'model/results.html', {'results': results.render_html()})
