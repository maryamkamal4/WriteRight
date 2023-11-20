from django.urls import path
from .views import image_comparison_view

urlpatterns = [
    path('image-comparison/', image_comparison_view, name='image_comparison'),
]