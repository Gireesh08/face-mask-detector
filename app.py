import streamlit as st
import cv2
import numpy as np
import os
import urllib.request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# auto-download face detector files if missing
if not os.path.exists("deploy.prototxt"):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "deploy.prototxt"
    )

if not os.path.exists("res10_300x300_ssd_iter_140000.caffemodel"):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "res10_300x300_ssd_iter_140000.caffemodel"
    )

# load model + face detector once
model = load_model("face_mask_detector.h5")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

st.title("😷 Face Mask Detector")

# initialize history storage (only runs once)
if "history" not in st.session_state:
    st.session_state.history = []

# function that runs the full detect + classify pipeline on one image
def process_image(img):
    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")

            face_crop = img[y1:y2, x1:x2]
            face_resized = cv2.resize(face_crop, (224, 224))
            face_array = face_resized.astype("float32")
            face_array = preprocess_input(face_array)
            face_array = face_array.reshape(1, 224, 224, 3)

            prediction = model.predict(face_array)[0][0]
            label = "Mask" if prediction < 0.5 else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} ({prediction:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return img

# let user choose input method
option = st.radio("Choose input method:", ["Upload Photo", "Use Camera"])

if option == "Upload Photo":
    input_image = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
else:
    input_image = st.camera_input("Take a photo")

if input_image is not None:
    file_bytes = np.asarray(bytearray(input_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    result_img = process_image(img)

    # save this result into history
    st.session_state.history.append(result_img)

# button to clear history if it gets too long
if st.session_state.history:
    if st.button("Clear History"):
        st.session_state.history = []

# display ALL results, most recent first
st.subheader("Detection Results")
for past_img in reversed(st.session_state.history):
    st.image(cv2.cvtColor(past_img, cv2.COLOR_BGR2RGB))