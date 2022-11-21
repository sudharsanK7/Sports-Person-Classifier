#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
from PIL import Image, ImageOps
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import pickle
import import_ipynb
import pywt
import json
import imageio.v2 as imageio
import base64
import pandas as pd


# In[2]:


face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')


# In[3]:


pik = open("./classifier.pkl","rb")
classifier = pickle.load(pik)


# In[4]:


def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor(imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H


# In[5]:


with open('./class_dictionary.json') as json_file:
    class_dict = json.load(json_file) 
class_dict2 = {value:key for key, value in class_dict.items()}


# In[6]:


RMD = {
    'Name' : ['david_alaba','fede_valverde','karim_benzema','luka_modric','toni_kroos'],
    'Goals' : ['1','0', '15', '0', '2'],
    'Assist' : ['1','1', '1', '4', '0']
}

df = pd.DataFrame(RMD)
dft = df.T


# In[7]:


def image_opening(u8):
    my_string = base64.b64encode(u8)
    return my_string


# In[8]:


def get_cv2_image_from_base64_string(b64str):
    nparr = np.frombuffer(base64.b64decode(b64str), dtype='B')
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


# In[9]:


def get_cropped_image_if_2_eyes(image_path):
    gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image_path[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color


# In[10]:


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


# In[11]:


def finale(strin):
    image_in = get_cv2_image_from_base64_string(strin)
    resultim = increase_brightness(image_in, value=20)
    imag = get_cropped_image_if_2_eyes(resultim)
    scalled_raw_img = cv2.resize(imag, (32, 32))
    img_har = w2d(imag,'db1',5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
    restacked_img = np.array(combined_img).reshape(1,4096).astype(float)
    prediction=classifier.predict(restacked_img)
    predi = (int(prediction))
    return predi


# In[12]:


def main():
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/56/Real_Madrid_CF.svg/800px-Real_Madrid_CF.svg.png", width=100)
    st.title("Real Madrid UCL 21-22 Players Classification")
    html_temp = """
    <div style="background-color:white;padding:10px">
    <h2 style="color:gold;text-align:center;">Know Your Players </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    file_in = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
    if file_in is None:
        st.text("Please upload an image file")
    else:
        bytes_data = file_in.getvalue()
        file64 = image_opening(bytes_data)
        result=""
    if st.button("Predict"):
        result=finale(file64)
        if result is not None:
            st.dataframe(dft[:][result])
        else:
            st.error('Upload some other photo')


if __name__ == '__main__':
    main()


# In[ ]:




