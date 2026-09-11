from django.urls import path
from . import views

app_name = 'project1'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_data, name='upload'),
    path('train/', views.train_model, name='train'),
]