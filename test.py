import tempfile
import streamlit as st
import numpy as np
from tensorflow import keras
import cv2
import io
import os
from PIL import Image
import tempfile

def load_image():
    uploaded_file = st.file_uploader(label='Drag Or Upload Your  x-ray image')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data, width=200)
        return Image.open(io.BytesIO(image_data))
    else:
        return None



def predict():
    model = keras.models.load_model("./mymodel.h5")
    data=[]
    path="positive.jpeg"
    path1="negative.jpeg"
   
    image = cv2.imread(path1)
    image = cv2.resize(image, (150,150))
    image = image.astype('float32')
    image /= 255
    data.append(image)
    data = np.array(data)
    predicted_labels = model.predict(data)
    predicted_labels = np.argmax(predicted_labels,axis=1)
    output = predicted_labels[0]
    if output==0:
	    st.success(' your covid test is Negative 🙂🙂')
    else:
	    st.error("  your covid test is Positive 😞😞")


def main():
    st.title("Welcome to E-CoralLab  ")
    st.text("Digital Testing Lab ")
    st.title('Upload your Lungs X-ray Image')
    image=load_image()
    result = st.button('Submit X-ray')
    if result:
        st.write('Loading into database Getting your Result ready..........')
        predict()
        


if __name__ == '__main__':
    main()
