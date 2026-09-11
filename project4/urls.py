from django.urls import path

app_name = 'project4'

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('study/', views.study_view, name='study'),
    path('pairwise/', views.pairwise_view, name='pairwise'),
    path('ranking/', views.ranking_view, name='ranking'),
    path('submit_pairwise/', views.submit_pairwise, name='submit_pairwise'),
    path('submit_ranking/', views.submit_ranking, name='submit_ranking'),
    path('results/', views.results_view, name='results'),
    path('report/', views.download_report, name='report'),
]