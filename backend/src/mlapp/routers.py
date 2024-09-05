from fastapi import APIRouter, Depends, status
from fastapi import FastAPI, File, UploadFile
from tempfile import NamedTemporaryFile
from fastapi.concurrency import run_in_threadpool
import os
import cv2
from os.path import dirname, join
import uuid
import cv2
from PIL import Image as pImage
import face_recognition

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import pandas as pd

MODEL_PATH = join(dirname(dirname(dirname(__file__))),'Meso4_DF.h5')
image_dimensions = {'height':256, 'width':256, 'channels':3}
# Create a Classifier class

class Classifier:
    def __init__(self):
        self.model = 0

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)


class Meso4(Classifier):
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate = learning_rate)
        self.model.compile(optimizer = optimizer,
                           loss = 'mean_squared_error',
                           metrics = ['accuracy'])

    def init_model(self):
        x = Input(shape = (image_dimensions['height'],
                           image_dimensions['width'],
                           image_dimensions['channels']))

        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return Model(inputs = x, outputs = y)

meso = Meso4()
meso.load(MODEL_PATH)


current_video_link = ""
ml_router =  APIRouter()


VIDEO_DIR= join(dirname(dirname(dirname(__file__))),'videos')
IMAGE_DIR= join(dirname(dirname(dirname(__file__))),'images')
preprocessed_images = []
cropped_images = []
frames = []
predictions = []

def process_video(filename):
    cap = cv2.VideoCapture(filename)

    #find fill uuid
    filearr = filename.split("/")#for windows change this
    fileuuid = filearr[-1]
    fileuuid = fileuuid.split('.')[0]

    #output video
    ouput_video = f'{fileuuid}_framed_.mp4'
    output_path = os.path.join(VIDEO_DIR, ouput_video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    forcc = cv2.VideoWriter_fourcc('M','P','4','V')

    out = cv2.VideoWriter(output_path,forcc,fps,(width,height))


    #ouptut text constants
    text = "Confidence: "
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 1
    font_color = (0, 0, 255)
    font_thickness = 2
    

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else: 
            break
    cap.release()

    print(f"Number of frames {len(frames)}")

    padding = 40
    faces_found = 0
    sequence_length = 50
    
    
    face_locs = []


    #preprocess  and crop images plus get face_loc
    for i in range(sequence_length):
        if i >= len(frames):
            break
        frame = frames[i]


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_name = f'{fileuuid}_preprocessed_{i+1}.png'
        image_path = os.path.join(IMAGE_DIR, image_name )
        img_rgb = pImage.fromarray(rgb_frame, 'RGB')
        img_rgb.save(image_path)       
        preprocessed_images.append(image_name)

        face_loc = face_recognition.face_locations(rgb_frame)
        face_locs.append(face_loc)

        if(len(face_loc) == 0):
            continue
        top, right, bottom, left = face_loc[0]


        face_frame = frame[top - padding:bottom + padding, left - padding: right + padding]
        face_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_image_rgb = pImage.fromarray(face_rgb, 'RGB')
        cropped_image_name = f'{fileuuid}_cropped_{i+1}.png'
        cropped_image_path = os.path.join(IMAGE_DIR, cropped_image_name)
        face_image_rgb.save(cropped_image_path)
        faces_found +=1
        cropped_images.append(cropped_image_name)

# Convert PIL Image to NumPy array
        face_image_np = np.array(face_image_rgb)

# Resize the image to the model's expected input size (256x256 in this case)
        face_image_np_resized = cv2.resize(face_image_np, (image_dimensions['width'], image_dimensions['height']))

# Normalize the image (assuming pixel values between 0 and 255, we normalize them to [0, 1])
        face_image_np_resized = face_image_np_resized / 255.0

# Expand dimensions to match the model input (batch_size, height, width, channels)
        face_image_np_resized = np.expand_dims(face_image_np_resized, axis=0)

# Make a prediction using the Meso4 model
        prediction = meso.predict(face_image_np_resized)

# Append the prediction to the predictions list
        predictions.append(prediction[0][0])
        print(i)
        print(prediction[0])

        color = (0, 0, 255)
        thickness = 2
        cv2.rectangle(frame, (left,top),(right,bottom),color, thickness)
        cv2.putText(frame,f"{prediction[0]}",(left, bottom),font,font_scale,font_color,font_thickness)
        out.write(frame)

    print(predictions)

    #get the video with drawn fram
    # cap = cv2.VideoCapture(filename)
    # ouput_video = f'{fileuuid}_framed_.mp4'
    # output_path = os.path.join(VIDEO_DIR, ouput_video)

    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)

    # forcc = cv2.VideoWriter_fourcc('M','P','4','V')

    # out = cv2.VideoWriter(output_path,forcc,fps,(width,height))

    # i = 0

    # while cap.isOpened():
    #     ret, frame = cap.read()

    #     if(i > sequence_length):
    #         break

    #     if ret:
    #         color = (0, 255, 0)
    #         thickness = 2
    #         cv2.rectangle(frame, face_locs[i][0],face_locs[i][4],color, thickness)
    #         i = i + 1;
    #         out.write(frame)
    #         cv2.imshow('video with rectangle  and text', frame)

    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     else: 
    #         break
        
    # cap.release()
    # cap.release()
    # cv2.destroyAllWindows()


process_video("/home/taichikarna/Sih/backend/videos/zuckerberfake.mp4")

@ml_router.post("/upload", status_code=status.HTTP_201_CREATED)
async def detect_faces(file: UploadFile ):
    try:
        content = await file.read()
        filename = f'{str(uuid.uuid4())}.mp4'
        current_video_link = os.path.join(VIDEO_DIR,filename)
        with open(os.path.join(VIDEO_DIR, filename ), 'wb+') as video:
            video.write(content)
        return {"message": "File uploaded successfully"}
    finally:
        return {"message": "failed"}
