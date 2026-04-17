from rest_framework import serializers
class BugSerializer(serializers.Serializer):
    text=serializers.CharField()
    language=serializers.CharField(default='python')

    def validate_text(self, value):
        if len(value.strip()) < 10:
            raise serializers.ValidationError("Please provide a proper error message")
        return value
