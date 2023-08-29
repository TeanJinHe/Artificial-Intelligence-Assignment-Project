import tkinter as tk
from tkinter import *
from tkinter import PhotoImage
from tkinter import Frame
from tkinter import Label
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import imutils
import time

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

def detect_mask_photo():
    file_path = filedialog.askopenfilename(title="Select Photo")
    if file_path:
        image = cv2.imread(file_path)

        prototxtPath = "face_detector\\deploy.prototxt"
        weightsPath = "face_detector\\res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        maskNet = load_model("mask_detector.model")

        (locs, preds) = detect_and_predict_mask(image, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        image.show()

def detect_mask_video():
    file_path = filedialog.askopenfilename(title="Select Video")
    if file_path:
        prototxtPath = "face_detector\\deploy.prototxt"
        weightsPath = "face_detector\\res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        maskNet = load_model("mask_detector.model")

        video_capture = cv2.VideoCapture(file_path)
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            cv2.imshow("Mask Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

def detect_mask_streaming():
    prototxtPath = "face_detector\\deploy.prototxt"
    weightsPath = "face_detector\\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    maskNet = load_model("mask_detector.model")

    video_capture = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Mask Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

window = tk.Tk()

image1 = PhotoImage(file="C:\\Users\\user\\OneDrive\\Desktop\\AI lecturer learning\\assignment\\chandrikadeb_Face_Mask_Detection\\mask1.png")
background_label = Label(window, image=image1)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


window.title("Mask Detection system")
window.geometry("800x600")

headingFrame1 = Frame(window,bg="#5ce1e6",bd=5)
headingFrame1.place(relx=0.05,rely=0.1,relwidth=0.9,relheight=0.16)
headingLabel = Label(headingFrame1, text="Mask Detection system", bg='black', fg='white', font=('Courier',18))
headingLabel.place(relx=0,rely=0, relwidth=1, relheight=1)


Detect_Mask_in_Photo_frame = Frame(window, bg='#a9d696', bd=5)
Detect_Mask_in_Photo_frame.place(relx=0.3, rely=0.35, relwidth=0.4, relheight=0.14)
Detect_Mask_in_Photo_Button = Button(Detect_Mask_in_Photo_frame, text="Detect_Mask_in_Photo", bg='#39382e', fg='white', font=('Courier', 15,'bold'), command=detect_mask_photo)
Detect_Mask_in_Photo_Button.place(relx=0, rely=0, relwidth=1, relheight=1)

Detect_Mask_in_Video_frame = Frame(window, bg='#a9d696', bd=5)
Detect_Mask_in_Video_frame.place(relx=0.3, rely=0.5, relwidth=0.4, relheight=0.14)
Detect_Mask_in_Video_Button = Button(Detect_Mask_in_Video_frame, text="Detect_Mask_in_Video", bg='#39382e', fg='white', font=('Courier', 15,'bold'), command=detect_mask_video)
Detect_Mask_in_Video_Button.place(relx=0, rely=0, relwidth=1, relheight=1)

Detect_Mask_Streaming_frame = Frame(window, bg='#a9d696', bd=5)
Detect_Mask_Streaming_frame.place(relx=0.3, rely=0.65, relwidth=0.4, relheight=0.14)
Detect_Mask_in_Streaming_Button = Button(Detect_Mask_Streaming_frame, text="Detect_Mask_Streaming", bg='#39382e', fg='white', font=('Courier', 15,'bold'), command=detect_mask_streaming)
Detect_Mask_in_Streaming_Button.place(relx=0, rely=0, relwidth=1, relheight=1)

Exit_frame = Frame(window, bg='#a9d696', bd=5)
Exit_frame.place(relx=0.35, rely=0.82, relwidth=0.3, relheight=0.10)
Exit_Button = Button(Exit_frame, text="Exit", bg='#39382e', fg='white', font=('Courier', 15,'bold'), command=window.quit)
Exit_Button.place(relx=0, rely=0, relwidth=1, relheight=1)



window.mainloop()