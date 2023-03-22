from flask import Flask, render_template, Response
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from cvzone.ClassificationModule import Classifier

app = Flask(__name__)
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
image_size = 224
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


@app.route('/')
def index():
    return render_template('index.html')


def generate_frames():
    while True:
        success, img = cap.read()
        img_output = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]

            img_white = np.ones((image_size, image_size, 3), np.uint8) * 255
            img_crop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            img_crop_shape = img_crop.shape

            aspect_ratio = h / w
            if aspect_ratio > 1:
                k = image_size / h
                calculated_w = math.ceil(k * w)
                img_resize = cv2.resize(img_crop, (calculated_w, image_size))
                img_resize_shape = img_resize.shape
                width_gap = math.ceil((image_size - calculated_w) / 2)
                img_white[:, width_gap:calculated_w + width_gap] = img_resize
                prediction, index = classifier.getPrediction(img_white, draw=False)
            else:
                k = image_size / w
                calculated_h = math.ceil(k * h)
                img_resize = cv2.resize(img_crop, (image_size, calculated_h))
                img_resize_shape = img_resize.shape
                height_gap = math.ceil((image_size - calculated_h) / 2)
                img_white[height_gap:calculated_h + height_gap, :] = img_resize
                prediction, index = classifier.getPrediction(img_white, draw=False)

            cv2.rectangle(img_output, (x - offset, y - offset - 50), (x - offset + 90, y - offset), (255, 0, 255),
                          cv2.FILLED)
            cv2.putText(img_output, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
            cv2.rectangle(img_output, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

            ret, buffer = cv2.imencode('.jpg', img_output)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
