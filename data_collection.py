import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

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
        cv2.imshow("Image Crop", img_crop)
        cv2.imshow("Image White", img_white)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
