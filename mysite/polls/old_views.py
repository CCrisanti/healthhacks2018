from django.shortcuts import render
from django.http import HttpResponse

def index(request):
	print("test")
	return HttpResponse("<b>Hello, world. You're at the polls index.")

# Create your views here.
