from rest_framework import serializers
from .models import BugLog
class BugSerializer(serializers.Serializer):
    text=serializers.CharField()
    language=serializers.CharField(default='python')

    def validate_text(self, value):
        if len(value.strip()) < 10:
            raise serializers.ValidationError("Please provide a proper error message")
        return value
# OUTPUT — formats what API returns for bug history/detail
class BugLogSerializer(serializers.ModelSerializer):
    class Meta:
        model  = BugLog
        fields = [
            'id', 'error_message', 'language', 'category',
            'confidence', 'method', 'concept_name', 'explanation',
            'pattern_detected', 'pattern_message', 'created_at'
        ]