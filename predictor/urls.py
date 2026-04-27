from django.urls import path
from . import views

urlpatterns = [
    path('login/',    views.login_view,  name='login'),
    path('logout/',   views.logout_view, name='logout'),
    path('register/', views.register,    name='register'),
    path('', views.home, name='home'),
    path('apply-seed/',      views.apply_seed,      name='apply_seed'),
    path('my-applications/', views.my_applications, name='my_applications'),
    path('predict/',         views.predict_price,   name='predict_price'),
    path('predict/export/<int:pk>/', views.export_prediction_pdf, name='export_pdf'),
    path('agent/',      views.agent_page, name='agent'),
    path('agent/chat/', views.agent_chat, name='agent_chat'),
]
