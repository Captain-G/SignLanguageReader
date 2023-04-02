import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
image_size = 224
counter = 0

folder = "Data/TEST"

while True:
    try:
        success, img = cap.read()
        # cap.read() returns either true or false(stored in success) , and an array of 0-255 stored as img
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            # hand contains a dictionary with the following  keys : bbox, center etc
            x, y, w, h = hand["bbox"]
            # x is the starting x co-oridnate, y is the starting y co-ordinate, w is the width, h is the height
            img_white = np.ones((image_size, image_size, 3), np.uint8) * 255
            # np.ones() creates a numpy array of size 224*224*3 of values initialized as 1s(black) *255 white
            img_crop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            # crops the img from the webcam footage to only the hand
            img_crop_shape = img_crop.shape
            # img_crop_shape returns dimensions of the cropped image eg 255, 204,3
            aspect_ratio = w / h
            if aspect_ratio > 1:
                k = image_size / w
                calculated_h = math.ceil(k * h)
                img_resize = cv2.resize(img_crop, (image_size, calculated_h))
                img_resize_shape = img_resize.shape
                height_gap = math.ceil((image_size - calculated_h) / 2)
                img_white[height_gap:calculated_h + height_gap, :] = img_resize
            else:
                k = image_size / h
                calculated_w = math.ceil(k * w)
                img_resize = cv2.resize(img_crop, (calculated_w, image_size))
                img_resize_shape = img_resize.shape
                width_gap = math.ceil((image_size - calculated_w) / 2)
                img_white[:, width_gap:calculated_w + width_gap] = img_resize
            cv2.imshow("Image Crop", img_crop)
            cv2.imshow("Image White", img_white)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", img_white)
            print(counter)
    except Exception as e:
        print(e)
