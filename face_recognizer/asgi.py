# myproject/asgi.py
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import analyser.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_recognizer.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            analyser.routing.websocket_urlpatterns
        )
    ),
})
# This file is the ASGI configuration for the Django project.
# It exposes the ASGI callable as a module-level variable named `application`.