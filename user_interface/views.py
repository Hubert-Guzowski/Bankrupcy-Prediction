from django.shortcuts import render


def homepage(request):
    result = ""
    if request.method == 'POST':

        result = """run_analysis(
            request.form['statistic-type'], request.form['social-group'],
            request.form['date-from'] or None, request.form['date-to'] or None
        )"""
    return render(request, 'interface/homepage.html', {'results': result})
