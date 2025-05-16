import os
import cv2
import dlib
import math
import json
import statistics
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import csv
from collections import deque
import tensorflow as tf
import sys
from tensorflow.keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO logs, but keeps warnings and errors
import tensorflow as tf


TOTAL_FRAMES = 22
LIP_WIDTH = 80
LIP_HEIGHT = 112
PAST_BUFFER_SIZE = 4
VALID_WORD_THRESHOLD = 1
NOT_TALKING_THRESHOLD = 2


INPUT_SHAPE = (TOTAL_FRAMES, LIP_WIDTH, LIP_HEIGHT, 3)
label_dict = {6: 'hello', 5: 'dog', 10: 'my', 12: 'you', 9: 'lips', 3: 'cat', 11: 'read', 0: 'a', 4: 'demo', 7: 'here', 8: 'is', 1: 'bye', 2: 'can'}
count = 0
#label_dict = {2: 'my', 1: 'lips', 3: 'read', 0: 'demo'}
LABEL_DICT = {6: 'hello', 5: 'dog', 10: 'my', 12: 'you', 9: 'lips', 3: 'cat', 11: 'read', 0: 'a', 4: 'demo', 7: 'here', 8: 'is', 1: 'bye', 2: 'can'}

# Define the input shape
input_shape = (TOTAL_FRAMES, 80, 112, 3)


def build_model():
    model = tf.keras.Sequential([
        Conv3D(32, (3, 3, 3), activation='relu', input_shape=INPUT_SHAPE, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling3D((2, 2, 2)),
        
        Conv3D(64, (3, 3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling3D((2, 2, 2)),
        
        Conv3D(128, (3, 3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling3D((2, 2, 2)),
        
        GlobalAveragePooling3D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(LABEL_DICT), activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the model architecture
model = build_model()


model.load_weights('./model2.weights.h5', by_name=True)

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("./face_weights.dat")

# read the image
cap = cv2.VideoCapture(r'C:\Users\DELL\Downloads\1.avi')
#cap.set(cv2.CAP_PROP_FPS, 60)
curr_word_frames = []
not_talking_counter = 0



first_word = True
labels = []

past_word_frames = deque(maxlen=PAST_BUFFER_SIZE)

ending_buffer_size = 5

predicted_word_label = None
draw_prediction = False

spoken_already = []
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)
while True:
    red, frame = cap.read()
    if red:
        # Convert image into grayscale
        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

        # Use detector to find landmarks
        faces = detector(gray)
        
        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point

            # Create landmark object
            landmarks = predictor(image=gray, box=face)

            # Calculate the distance between the upper and lower lip landmarks
            mouth_top = (landmarks.part(51).x, landmarks.part(51).y)
            mouth_bottom = (landmarks.part(57).x, landmarks.part(57).y)
            lip_distance = math.hypot(mouth_bottom[0] - mouth_top[0], mouth_bottom[1] - mouth_top[1])



            lip_left = landmarks.part(48).x
            lip_right = landmarks.part(54).x
            lip_top = landmarks.part(50).y
            lip_bottom = landmarks.part(58).y

            # Add padding if necessary to get a 76x110 frame
            width_diff = LIP_WIDTH - (lip_right - lip_left)
            height_diff = LIP_HEIGHT - (lip_bottom - lip_top)
            pad_left = width_diff // 2
            pad_right = width_diff - pad_left
            pad_top = height_diff // 2
            pad_bottom = height_diff - pad_top

            # Ensure that the padding doesn't extend beyond the original frame
            pad_left = min(pad_left, lip_left)
            pad_right = min(pad_right, frame.shape[1] - lip_right)
            pad_top = min(pad_top, lip_top)
            pad_bottom = min(pad_bottom, frame.shape[0] - lip_bottom)

            # Create padded lip region
            lip_frame = frame[lip_top - pad_top:lip_bottom + pad_bottom, lip_left - pad_left:lip_right + pad_right]
            lip_frame = cv2.resize(lip_frame, (LIP_HEIGHT, LIP_WIDTH))  # Swap width and height

            
            lip_frame_lab = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2LAB)
            # Apply contrast stretching to the L channel of the LAB image
            l_channel, a_channel, b_channel = cv2.split(lip_frame_lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
            l_channel_eq = clahe.apply(l_channel)

            # Merge the equalized L channel with the original A and B channels
            lip_frame_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
            lip_frame_eq = cv2.cvtColor(lip_frame_eq, cv2.COLOR_LAB2BGR)
            lip_frame_eq= cv2.GaussianBlur(lip_frame_eq, (7, 7), 0)
            lip_frame_eq = cv2.bilateralFilter(lip_frame_eq, 5, 75, 75)
            kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])

            # Apply the kernel to the input image
            lip_frame_eq = cv2.filter2D(lip_frame_eq, -1, kernel)
            lip_frame_eq= cv2.GaussianBlur(lip_frame_eq, (5, 5), 0)
            lip_frame = lip_frame_eq
            
            
            # Draw a circle around the mouth
            for n in range(48, 61):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img=frame, center=(x, y), radius=1, color=(155, 255, 0), thickness=-1)
            
            if lip_distance > 11: # person is talking
                cv2.putText(frame, "Talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                curr_word_frames += [lip_frame.tolist()]
                not_talking_counter = 0
                draw_prediction = False
            else:
                cv2.putText(frame, "Not talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                not_talking_counter += 1
             
                if not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE == TOTAL_FRAMES: 

                    curr_word_frames = list(past_word_frames) + curr_word_frames

                    curr_data = np.array([curr_word_frames[:input_shape[0]]])

                    print("*********", curr_data.shape)
                    print(spoken_already)
                    prediction = model.predict(curr_data)

                    prob_per_class = []
                    for i in range(len(prediction[0])):
                        prob_per_class.append((prediction[0][i], label_dict[i]))
                    sorted_probs = sorted(prob_per_class, key=lambda x: x[0], reverse=True)
                    for prob, label in sorted_probs:
                        print(f"{label}: {prob:.3f}")

                    predicted_class_index = np.argmax(prediction)
                    while label_dict[predicted_class_index] in spoken_already:
                        # If the predicted label has already been spoken,
                        # set its probability to zero and choose the next highest probability
                        prediction[0][predicted_class_index] = 0
                        predicted_class_index = np.argmax(prediction)
                    predicted_word_label = label_dict[predicted_class_index]
                    spoken_already.append(predicted_word_label)

                    print("FINISHED!", predicted_word_label)
                    draw_prediction = True
                    count = 0

                    curr_word_frames = []
                    not_talking_counter = 0
                elif not_talking_counter < NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE < TOTAL_FRAMES and len(curr_word_frames) > VALID_WORD_THRESHOLD:
                    curr_word_frames += [lip_frame.tolist()]
                    not_talking_counter = 0
                elif len(curr_word_frames) < VALID_WORD_THRESHOLD or (not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE > TOTAL_FRAMES):
                    curr_word_frames = []

                past_word_frames+= [lip_frame.tolist()]
                if len(past_word_frames) > PAST_BUFFER_SIZE:
                    past_word_frames.pop(0)

        if(draw_prediction and count < 10):
            count += 1
            cv2.putText(frame, predicted_word_label, (50 ,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow(winname="Lip Reader", mat=frame)
        out.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            spoken_already = []

        # Exit when escape is pressed
        if key == 27:
            break
    else:
        break

cap.release()
out.release()
# Close all windows
cv2.destroyAllWindows()
