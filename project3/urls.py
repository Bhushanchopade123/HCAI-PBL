from django.urls import path

app_name = 'project3'

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('baseline/', views.baseline_view, name='baseline'),
    path('expert/', views.expert_view, name='expert'),
    path('defer/', views.defer_view, name='defer'),
    path('active_learning/', views.active_learning_view, name='active_learning'),
    path('report/', views.download_report, name='report'),
]