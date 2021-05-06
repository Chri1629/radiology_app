from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('home/', views.home, name='home'),
    path('index/', views.index, name='index'),
    path('upload/', views.upload, name='upload'),
    path('about/', views.about, name='about'),
    path('results/', views.results, name='results'),
    path('features/', views.features, name='features'),
    path('index2/', views.index2, name='index2'),

]