import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
import sklearn
from cv2 import cv2
from sklearn.decomposition import PCA
from PIL import Image


haar=cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
mean=pickle.load(open('./model/mean_preprocess.pickle','rb'))
model_svm=pickle.load(open('./model/model_svm.pickle','rb'))
model_pca=pickle.load(open('./model/pca_50.pickle','rb'))

print(' hurray')


gender_pre={0:'Male',1:'Female'}
font=cv2.FONT_HERSHEY_SIMPLEX   

def pipeline_model(path,filename,color='rgb'):
    # read image in cv2
    img=cv2.imread(path)
    if color=='bgr':
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


    faces=haar.detectMultiScale(gray,1.5,3)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        #Cropping
        cp=gray[y:y+h,x:x+w]
        #Normalisation
        cp=cp/255.0
        #Resizing image into (100,100)
        if cp.shape[1]>100:
            cp_resize=cv2.resize(cp,(100,100),cv2.INTER_AREA)
        else:
            cp_resize=cv2.resize(cp,(100,100),cv2.INTER_CUBIC)
        # Flattening 1x 10000

        cp_reshape=cp_resize.reshape(1,10000) # 1x 10000
        #plt.imshow(cp_reshape)
        #subtracting with mean
        cp_mean=cp_reshape-mean # saver is mean
        # Eigen image
        eigen_image=model_pca.transform(cp_mean)

        # Pass to ml Model(Support Vector Machine)
        results=model_svm.predict_proba(eigen_image)[0]

        # prediction
        predict=results.argmax() # 0 or 1
        score=results[predict]

        #sss
        text= "%s : %0.2f"%(gender_pre[predict],score)
        cv2.putText(img,text,(x,y),font,1,(0,255,255),3)
    cv2.imwrite('./static/prediction/{}'.format(filename),img)
