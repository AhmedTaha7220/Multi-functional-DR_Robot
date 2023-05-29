############################
############################
#Loading our models and prediction

#Initializing our libraries
import numpy as np
# this library is used to convert the images into numbers or matrices of numbers
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
# this library is used to draw graphs or plots
import os
# this library is used to read extensions from drivers inside operating system
import cv2
# this library is used to be able to import the dataset that you will use
from tqdm import tqdm
# this library is used to see the progress of reading data from your data set
import tensorflow as tf

#Brain tumor model and prediction
def brain_code(n):
    code = {0:'Meningioma' , 1:'Glioma' , 2:'Pituitary tumor'}
    for x,y in code.items():
        if n==x:
            return y

def brain():
    #Creating image path
    p=path.get()
    main_fold="images/"
    p=main_fold+p
    #Displaying the image
    img = PhotoImage(file=p)
    img=img.subsample(2,2)
    imagee = Label(root,image = img).place(x=0,y=100)
    #Applying the model on the image for prediction
    Brain_Model = tf.keras.models.load_model('Brain.model')
    y_img = []
    image1 = cv2.imread(p)
    image_array1 = cv2.resize(image1,(100,100))
    y_img.append(list(image_array1))
    y_img = np.array(y_img)
    y_result =  Brain_Model.predict(y_img)
    out['text']=brain_code(np.argmax(y_result))
    root.mainloop()
    
#Eyes model prediction
def eyes_code(n):
    code = {'Normal':0, 'Cataract':1, 'Glaucoma':2, 'Diabetic_retinopathy':3}
    for y,x in code.items():
        if n==x:
            return y

def eyes():
    #Creating image path
    p=path.get()
    main_fold="images/"
    p=main_fold+p
    #Displaying the image
    img = PhotoImage(file=p)
    img=img.subsample(2,2)
    imagee = Label(root,image = img).place(x=0,y=100)
    #Applying the model on the image for prediction
    Eyes_Model = tf.keras.models.load_model('Eyes.model')
    y_img = []
    image1 = cv2.imread(p)
    image_array1 = cv2.resize(image1,(100,100))
    y_img.append(list(image_array1))
    y_img = np.array(y_img)
    y_result =  Eyes_Model.predict(y_img)
    out['text']=eyes_code(np.argmax(y_result))
    root.mainloop()
    
#Chest model prediction
def chest_code(n):
    code_chest = {'NORMAL':0 ,'PNEUMONIA':1}
    for y,x in code_chest.items():
        if n==x:
            return y

def chest():
    #Creating image path
    p=path.get()
    main_fold="images/"
    p=main_fold+p
    #Displaying the image
    img = PhotoImage(file=p)
    img = img.subsample(6,6)
    imagee = Label(root,image = img).place(x=0,y=100)
    #Applying the model on the image for prediction
    Chest_Model = tf.keras.models.load_model('Chest.model')
    y_img = []
    image1 = cv2.imread(p)
    image_array1 = cv2.resize(image1,(224,224))
    y_img.append(list(image_array1))
    y_img = np.array(y_img)
    y_result =  Chest_Model.predict(y_img)
    out['text']=chest_code(np.argmax(y_result))
    root.mainloop()
    
#Covid model prediction
def covid_code(n):
    code={0:'COVID', 1:'Normal', 2:'Viral Pneumonia', 3:'Lung opacity' }
    for x,y in code.items():
        if n==x:
            return y

def covid():
    #Creating image path
    p=path.get()
    main_fold="images/"
    p=main_fold+p
    #Displaying the image
    img = PhotoImage(file=p)
    img=img.subsample(2,2)
    imagee = Label(root,image = img).place(x=0,y=100)
    #Applying the model on the image for prediction
    Covid_Model = tf.keras.models.load_model('Covid.model')
    y_img = []
    image1 = cv2.imread(p)
    image_array1 = cv2.resize(image1,(100,100))
    y_img.append(list(image_array1))
    y_img = np.array(y_img)
    y_result =  Covid_Model.predict(y_img)
    print(y_result)
    out['text']=covid_code(np.argmax(y_result))
    root.mainloop()
########################################################
########################################################
#Building Our GUI

from tkinter import *
root = Tk()
root.title("Neural Networks With Health")
root.geometry("1000x800")


#Adding Background image
from tkinter import ttk
img = PhotoImage(file="D:\\images\\Apps\\neural network2.png")
img=img.zoom(2,2)
lbl = Label(root,image = img).place(x=0,y=0)

#Adding The title of the app
title = Label(root, text="Dr:Robot",font=("Times New Roman",50),fg="white",bg='#00614c',pady=20)
title.pack()
#Adding the label and the entry that will holds our dataset path
#first adding label for descriping the entry
descrip= Label(root,text="Enter your image path here",font=("Arial",18),fg='white',bg='#008879')
descrip.place(x=20,y=200)
#second adding entry
path=StringVar()
path.set("img_name.png")
entr=Entry(root,font=("Arial",18),width=40,textvariable=path)
pa=path.get()
entr.place(x=400,y=200)



#Third displaying the predicated output
pred=Label(root,text="The predicated output is: ",font=("Arial",25),fg='#e89eb0',bg='#3f5256')
pred.place(x=20,y=450)
out=Label(root,text="",font=("Arial",50),fg='#e89eb0',bg='#3f5256')
out.place(x=400,y=450)

#Partioning our app to 3 splits
eyedis= Button(root,text="Eye diagnosis",font=("Arial",25),fg='white',bg='#2c908b',command=eyes)
eyedis.place(x=10,y=600)

headdis= Button(root,text="Brain diagnosis",font=("Arial",25),fg='white',bg='#2c908b',command=brain)
headdis.place(x=235,y=600)

Corona= Button(root,text="COVID diagnosis",font=("Arial",25),fg='white',bg='#2c908b',command=covid)
Corona.place(x=480,y=600)

Chest= Button(root,text="Chest diagnosis",font=("Arial",25),fg='white',bg='#2c908b',command=chest)
Chest.place(x=750,y=600)
root.mainloop()