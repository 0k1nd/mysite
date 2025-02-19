from rest_framework import serializers

class CountrySerializer(serializers.Serializer):
    country = serializers.CharField(max_length=100)