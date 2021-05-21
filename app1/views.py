from flask import render_template,request
from flask import redirect,url_for
import os
from PIL import Image
from app1.ml_model import pipeline_model

upload_folder='static/uploads'

def base():
    return render_template("base.html")

def index():
    return render_template("index.html")

def faceapp():
    return render_template("faceapp.html")

def get_width(path):
    img=Image.open(path)
    size=img.size
    aspect_ratio= size[0]/size[1]  # width/height
    w=250*aspect_ratio
    return int(w)



def gender():
    if request.method=='POST':
        f=request.files['image']
        filename=f.filename
        path=os.path.join(upload_folder,filename)
        f.save(path)
        w=get_width(path)
        # Predictions (Passing to pipeline model)
        pipeline_model(path,filename,color='bgr')
        return render_template("gender.html",fileupload=True,img=filename,w=w)

    return render_template("gender.html",fileupload=False,img="veoim.jpg", w='250')
