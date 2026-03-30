

# Create your models here.

from django.db import models
from django.contrib.auth.models import User

class BugLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    LANGUAGE_CHOICES = [
        ("python", "Python"),
        ("c", "C"),
        ("cpp", "C++"),
    ]

    error_message = models.TextField()
    language = models.CharField(max_length=10, choices=LANGUAGE_CHOICES, default="python")

    category = models.CharField(max_length=50)
    confidence = models.FloatField()
    method = models.CharField(max_length=20)

    concept_name = models.CharField(max_length=100)
    explanation = models.TextField()

    embedding = models.JSONField()

    pattern_detected = models.BooleanField(default=False)
    pattern_message = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.category} | {self.error_message[:30]}"