from django.shortcuts import render
#from tkinter.filedialog import askopenfilename
#from tkinter.filedialog import askdirectory
from django.shortcuts import render  
from django.http import HttpResponse  
from tac.functions import handle_uploaded_file
from tac.file import Test

def home(request):
    return render(request, 'base.html')

def index(request):
    if request.method == 'POST':  
        test = Test(request.POST, request.FILES)  
        if test.is_valid():  
            handle_uploaded_file(request.FILES['file'])  
            variables = {"status":"correct", "name": request.FILES['file'].name}
            print("AAAAA", variables["name"])
            return render(request,"index2.html",context=variables)
    else:  
        test = Test()  
        return render(request,"index.html", {'form':test})  

def index2(request, image):
    return render(request,"index2.html",context=image)

def upload(request):
    return render(request, 'upload.html')

def about(request):
    return render(request, 'about.html')

def results(request):
    return render(request, 'results.html')

def features(request):
    return render(request, 'features.html')