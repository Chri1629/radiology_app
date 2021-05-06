from django import forms  

class Test(forms.Form):  
    file = forms.FileField() # for creating file input