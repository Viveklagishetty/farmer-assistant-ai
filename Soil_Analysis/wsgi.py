"""
WSGI config for Agriculture_Soil_Analysis_And_Crop_Recommendation_using_ML project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Agriculture_Soil_Analysis_And_Crop_Recommendation_using_ML.settings')

application = get_wsgi_application()
