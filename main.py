import numpy as np
import cv2 
from scipy.stats import mode
import os
import csv
import time

path ='C:/Users/Notebook/Desktop/Projetos/Python Reconhecimento Facial'

os.chdir(path)

titulo= "Teste"+time.strftime("%Y-%m-%d")
saida= open('face_recon'+titulo+'.csv','w')
export = csv.writer(saida,quoting=csv.QUOTE_NONNUMERIC)

file_list = []
for file in os.listdir(path):
    if file.endswith(".jpg"):
        file_list.append(file)



for file in file_list:
    
    face_cascade = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    face_alt_cascade = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml')
    face_alt2_cascade = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    face_alt_tree_cascade = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml')

    img = cv2.imread(file)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    faces2 = face_alt_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    faces3 = face_alt2_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    faces4 = face_alt_tree_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

    classifiers = [faces, faces2, faces3, faces4]

    for classifier in classifiers:
        for (x,y,w,h) in classifier:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
        
        print("Para a imagem "+file+", foram encontradas {0} faces!".format(len(classifier)))


        encontrados = []
    
    for (classifier) in classifiers:
        x = format(len(classifier))
        encontrados.append(x)
    
    encontrados = np.asarray(encontrados, dtype = np.float16)
    media = np.mean(encontrados)
    variancia = np.var(encontrados)
    moda = float(mode(encontrados)[0])
    
    if file == file_list[0]:
        export.writerow(["imagem","media","variancia","moda"])
        export.writerow([file, media, variancia, moda])
    else:
        export.writerow([file, media, variancia, moda])

saida.close()

print ('Fim!')