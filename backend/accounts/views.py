from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from .serializers import RegisterSerializer
from rest_framework import status
@api_view(['POST'])
@permission_classes([AllowAny])
def register_view(request):
    serializer=RegisterSerializer(data=request.data)

    if serializer.is_valid():
        serializer.save()
        return Response( {'message': 'User registered successfully'}, ## response is auto detects and send if html or json auto detect content to be browsable api
            status=status.HTTP_201_CREATED
        )
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



from .serializers import OnboardingSerializer

@api_view(['POST'])
@permission_classes([IsAuthenticated])  # must be logged in
def onboarding_view(request):
    serializer = OnboardingSerializer(data=request.data)
    if serializer.is_valid():
        profile = request.user.profile   # access UserProfile via OneToOne
        profile.language   = serializer.validated_data['language']
        profile.experience = serializer.validated_data['experience']
        profile.save()
        return Response({'message': 'Onboarding complete'}, status=200)
    return Response(serializer.errors, status=400)

