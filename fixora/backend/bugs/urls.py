from django.urls import path,include
from bugs import views

urlpatterns = [
    
    path('analyze/',views.analyze_view, name='analyze')

]