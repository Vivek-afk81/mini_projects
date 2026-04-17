from django.urls import path,include
from bugs import views

urlpatterns = [
    
    path('analyze/',views.analyze_view, name='analyze'),
      path('bugs/',                 views.bug_history_view),       # NEW
    path('bugs/<int:pk>/',        views.bug_detail_view),        # NEW
    path('patterns/',             views.pattern_summary_view)  # NEW


]