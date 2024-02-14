# Take image input from system camera
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf
classes=['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']
test_set = tf.keras.utils.image_dataset_from_directory(
    r'C:\Users\ashut\ML Notes\6 Month internship AIT\Fruit_Detection',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

# def classify(frame):
#     model_path = r'C:\Users\ashut\ML Notes\6 Month internship AIT\Fruit_Detection\trained_model.h5'
#     model = load_model(model_path)
#     frame = cv2.resize(frame, (64, 64))
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     frame = image.img_to_array(frame)
#     frame = frame / 255.0
#     frame = frame.reshape(64, 64,3)
#     frame = np.expand_dims(frame, axis=0)
#     predictions = model.predict(frame)
#     # predicted_class_idx = np.argmax(predictions)
#     # print(predicted_class_idx)
#     # predicted_class = classes[predicted_class_idx]
#
#     # confidence = predictions[0][predicted_class_idx]
#     return np.argmax(predictions)

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model_tunned.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    img_array = np.array(img)
    model = r'C:\Users\ashut\ML Notes\6 Month internship AIT\Fruit_Detection\trained_model.h5'
    # result = classify(img_array)
    result=model_prediction(img_file_buffer)
    with open("labels.txt") as f:
        content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
    st.success("Model is Predicting it's a {}".format(label[result]))

    # st.text('Detected :',test_set.class_names[result])
