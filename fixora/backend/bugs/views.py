from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import BugSerializer
from .services.services import process_bug
from django.db.models import Count
from .models import BugLog
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from .serializers import BugLogSerializer

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


# ── GET /api/bugs/ ───────────────────────────────────────────────────────────
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def bug_history_view(request):
    bugs = BugLog.objects.filter(
        user=request.user
    ).order_by('-created_at')          # newest first

    serializer = BugLogSerializer(bugs, many=True)
    return Response(serializer.data)


# ── GET /api/bugs/<pk>/ ──────────────────────────────────────────────────────
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def bug_detail_view(request, pk):
    try:
        bug = BugLog.objects.get(pk=pk, user=request.user)
    except BugLog.DoesNotExist:
        return Response(
            {'error': 'Bug not found'},
            status=status.HTTP_404_NOT_FOUND
        )

    serializer = BugLogSerializer(bug)
    return Response(serializer.data)


# ── GET /api/patterns/ ───────────────────────────────────────────────────────
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def pattern_summary_view(request):
    bugs = BugLog.objects.filter(user=request.user)

    total = bugs.count()

    if total == 0:
        return Response({
            'total_bugs':  0,
            'streak':      request.user.profile.streak,
            'patterns':    [],
            'most_common': None
        })

    # group by category, count each, sort highest first
    grouped = (
        bugs
        .values('category')
        .annotate(count=Count('id'))
        .order_by('-count')
    )

    patterns = [
        {
            'category':   item['category'],
            'count':      item['count'],
            'percentage': round((item['count'] / total) * 100)
        }
        for item in grouped
    ]

    return Response({
        'total_bugs':  total,
        'streak':      request.user.profile.streak,
        'patterns':    patterns,
        'most_common': patterns[0]['category']  # highest count
    })
