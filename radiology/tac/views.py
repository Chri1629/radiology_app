from django.shortcuts import render

def home(request):
    return render(request, 'base.html')

def index(request):
    return render(request, 'index.html')

def upload(request):
    return render(request, 'upload.html')

def about(request):
    return render(request, 'about.html')

def results(request):
    return render(request, 'results.html')

def features(request):
    return render(request, 'features.html')