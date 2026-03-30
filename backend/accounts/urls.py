from django.urls import path,include
from accounts import views
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    
    path('register/',views.register_view, name='register'),
      path('login/', TokenObtainPairView.as_view()),
    path('refresh/', TokenRefreshView.as_view()),
#direclty handles login and refresh things login returns acces and refresh tokens which when verified gives user object in user in accounts/views function
]