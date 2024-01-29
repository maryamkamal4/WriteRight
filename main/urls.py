from django.urls import path
from .views import image_comparison_view

urlpatterns = [
    path('image_comparison', image_comparison_view, name='image_comparison'),
    # path('api/image1', image1_api, name='image1_api'),
    # path('api/image2', image2_api, name='image2_api'),
    # path('api/combined_images', combined_images_api, name='combined_images_api'),
]
