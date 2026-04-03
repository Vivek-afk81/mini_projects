from rest_framework import serializers
from django.contrib.auth.models import User
from .models import UserProfile
class RegisterSerializer(serializers.ModelSerializer):
    password=serializers.CharField(write_only=True)  # write only true input comes password but not included in output response

    class Meta:
        fields=['username','email','password']
        model=User
    

    def create(self,validated_data): # to save hashed passaword auto to db is runeed when serailizer.save runs
        #validate data
        username=validated_data['username']
        password=validated_data['password']
        email=validated_data.get('email','') #may be optional 
        # create user and save
        user=User.objects.create_user(
            username=username,
            password=password,
            email=email


        )
        return user


class OnboardingSerializer(serializers.Serializer):
    language   = serializers.ChoiceField(choices=['python', 'c', 'cpp'])
    experience = serializers.ChoiceField(choices=['beginner', 'intermediate'])

#view calls this




