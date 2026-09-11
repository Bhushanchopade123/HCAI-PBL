from django.urls import path

app_name = 'project2'

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('tree/', views.tree_view, name='tree'),
    path('logistic/', views.logistic_view, name='logistic'),
    path('counterfactual/', views.counterfactual_view, name='counterfactual'),
    path('feature_effects/', views.feature_effects_view, name='feature_effects'),
]