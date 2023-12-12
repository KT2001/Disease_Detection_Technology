# import system libs
import os
import time
import shutil
import pathlib
import itertools

# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import streamlit as st
import PIL
from PIL import Image, ImageOps
st.set_option('deprecation.showfileUploaderEncoding', False)
#@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('/content/drive/MyDrive/DDT/Models/ResNet50/resnet50-Disease-93.25.h5')
  model.load_weights('/content/drive/MyDrive/DDT/Models/ResNet50/resnet50-Disease-weights.h5')
  model.summary()
  return model
def image_loader(file):
  # Load the image using PIL
  img = image.load_img(file, target_size=(224, 224))
  # Convert the image to a numpy array
  img_array = image.img_to_array(img)
  # Add an extra dimension to represent the batch size
  img_array = np.expand_dims(img_array, axis=0)

  # Preprocess the image
  preprocessed_img = preprocess_input(img_array)

  return preprocessed_img
def load_and_predict(image_reshaped, model):
  #image_reshaped = image[np.newaxis, ...]
  prediction = model.predict(image_reshaped)

  return prediction
def main():
  st.title("Disease Detection Based on Radiology Scans")
  file = st.file_uploader("Please upload the Radiology scan image", type=["png", "jpg", "jpeg"])
  if file is not None:
    image = Image.open(file)
    st.image(image, caption="Your image", use_column_width=True)
    img = image_loader(file)
    model = load_model()
    pred = load_and_predict(img, model)
    class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    st.write("Your image is of disease")
    st.success(class_names[np.argmax(pred)])
  else:
    print("please upload a proper file")

main()
