from django.db import models

class ImageModel(models.Model):
    title = models.CharField(max_length=100)
    image = models.ImageField(upload_to='images/')
    class Meta:
        app_label = 'myapp'
class Hello(models.Model):
    name=models.CharField(max_length=10)