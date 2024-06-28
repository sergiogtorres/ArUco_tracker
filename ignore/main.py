import cv2
import numpy as np

import time

#TO ADD
# calculate distances to QR codes with stereo cameras
# calculate distances to QR codes with mono cameras, knowing size of QR? QR represents size!!
# ^first calibrate camera... calibrate it automatically in real time?

CAP_WIDTH, CAP_HEIGHT = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

detector = cv2.QRCodeDetector()


while cap.isOpened():

    success, img = cap.read()
    start = time.perf_counter()

    value, points, qrcode = detector.detectAndDecode(img)

    if value != "":

        x1 = points[0][0][0]
        y1 = points[0][0][1]
        x2 = points[0][2][0]
        y2 = points[0][2][1]

        x_center = (x2 - x1) / 2 + x1
        y_center = (y2 - y1) / 2 + y1

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0))

        cv2.circle(img, (int(x_center), int(y_center)), 3, (0, 0, 255), 3)

        cv2.putText(
            img,  # Image
            str(value),  # Text to display
            (30, 120),  # Bottom-left corner of the text string in the image
            cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            1,  # Font scale (size of the text)
            (255, 255, 255),  # Color (white in BGR)
            2  # Thickness of the text
        )

    end = time.perf_counter()
    totalTime = end - start
    fps = 1 / totalTime

    cv2.putText(img, f'FPS: {int(fps)}', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

cv2.destroyAllWindows()