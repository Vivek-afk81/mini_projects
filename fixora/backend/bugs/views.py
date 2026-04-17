from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import BugSerializer
from .services.services import process_bug
# Create your views here.

@api_view(['POST'])
def analyze_view(request):
    serializer=BugSerializer(data=request.data)


    if serializer.is_valid():
        user=request.user
        text=serializer.validated_data['text']
        language=serializer.validated_data['language']
        
        result=process_bug(user,text,language)
        request.user.profile.update_streak() 
        return Response(result)
    return Response(serializer.errors, status=400)


