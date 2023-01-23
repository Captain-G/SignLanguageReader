import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
image_size = 300

while True:
    success, img = cap.read()
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
        else:
            k = image_size / w
            calculated_h = math.ceil(k * h)
            img_resize = cv2.resize(img_crop, (image_size, calculated_h))
            img_resize_shape = img_resize.shape
            height_gap = math.ceil((image_size - calculated_h) / 2)
            img_white[height_gap:calculated_h + height_gap, :] = img_resize

        cv2.imshow("Image Crop", img_crop)
        cv2.imshow("Image White", img_white)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
