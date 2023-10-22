from django.contrib import admin
from django.urls import path,include
from . import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    # path('',views.index,name='index'),
    # path('',views.upload_image,name="uploadimage"),
    path('',views.main,name="main"),
    path('home',views.home,name="home"),
    path('find',views.find,name="find"),
    path('about',views.about,name="about"),
    path('contact',views.contact,name="contact"),
    path('checkeligibility',views.checkeligibility,name="checkeligibility"),

]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
