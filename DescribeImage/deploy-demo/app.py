import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow
from tensorflow.keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import cv2
from numpy import argmax,array 
from pickle import load

image_preprocess = load_model("preprocess.h5")
final_model = load_model("model_9.h5")
t = open("tokenize.pkl","rb")
tokenize = load(t)
m = open("maxlength.dump","rb")
max_length = load(m)

def load_image(image_file):
    image = Image.open(image_file)
    return image

def word_for_id(integer,tokenizer):
    for word,index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model,tokenizer,photo, max_length):
    photo = image_preprocess.predict(photo)
    in_text = "<start>"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence],maxlen=max_length)
        yhat = model.predict([photo,sequence],verbose=0)

        yhat = argmax(yhat)
        word = word_for_id(yhat,tokenizer)
        if word is None:
            break
        in_text += " "+word
        if word == "end":
            break
    return in_text

def main():
    st.title("Welcome to Auto Image Caption Generator")
    menu = ["Image", "About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Image":
        st.subheader("Upload an image to generate the captions")
        image_file = st.file_uploader("Upload Images", type = ["png","jpg","jpeg"])
        if image_file is not None:
            file_details = {"filename":image_file.name,
            "filetype":image_file.type,
            "filesize":image_file.size
            }
            st.write(file_details)
            st.image(load_image(image_file))

            img_arr = np.array(load_image(image_file))
            new_image = img_arr.copy()
            pre_step1 = cv2.resize(new_image, (224,224))
            image_array = tf.keras.preprocessing.image.img_to_array(pre_step1)
            image_array = tf.expand_dims(image_array,0)
            image_array = preprocess_input(image_array)
            yhat = generate_desc(final_model,tokenize,image_array,max_length)
            st.write(yhat)
    
    elif (choice == "About"):
        st.subheader("About Project")

if __name__ == "__main__":
    main()