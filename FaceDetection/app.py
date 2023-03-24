import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow
import cv2
import numpy as np
from tensorflow.keras.models import load_model

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
detector = load_model("dl-model.save")

def load_image(image_file):
    img = Image.open(image_file)
    return img

def main():
    st.title("Welcome to Covid 19 - Face Mask Detector")
    menu = ["Detector", "About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Detector":
        st.subheader("Upload an Image to check Violoation")
        image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
        if image_file is not None:
            file_details = {"filename" : image_file.name, "filetype":image_file.type,
            "filesize": image_file.size
            }
            st.write(file_details)
        image_array = np.array(load_image(image_file))
        new_image = image_array.copy()
        new_image = cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(new_image, 1.1, 4)  

        for x,y,w,h in faces:
            face_img = image_array[y:y+h,x:x+w] # get face crop
            resized = cv2.resize(face_img, (224,224))
            image_arr = tf.keras.preprocessing.image.img_to_array(resized)
            image_arr = tf.expand_dims(image_arr,0)
            predictions = detector.predict(image_arr)
            score = tf.nn.softmax(predictions[0])
            label = np.argmax(score) 
        
        new_image_2 = image_array.copy()
        if label == 0 :
            cv2.rectangle(new_image_2,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(new_image_2,"mask", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8 , (0,255,0), 2, cv2.LINE_AA)
        
        if label == 1:
            cv2.rectangle(new_image_2,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(new_image_2,"no_mask", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8 , (0,0,255), 2, cv2.LINE_AA)
        
        cv2.imwrite("out.jpg",new_image_2)
        st.image(load_image("out.jpg"))



    elif choice == "About":
        st.subheader("About Project")


if __name__ == "__main__":
    main()