from django.http import HttpResponse

from bankruptcy_model.utils import data_loading


def dataframe_view(request, year_number):
    dataframe = data_loading.load_dataset_by_year(year_number)
    return HttpResponse(dataframe.to_html())
