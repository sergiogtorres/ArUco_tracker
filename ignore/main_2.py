import cv2
import numpy as np

import time
from utils import Perspective
#TO ADD
# NOT NECESSARY, already absolute scale. calculate distances to QR codes with stereo cameras
# DONE with CV2 algo. calculate distances to QR codes with mono cameras, knowing size of QR? QR represents size!!
# DONE with Nivolai Nielssen code ^first calibrate camera... calibrate it automatically in real time?
# Use Kalman Filter to improve noisy readings of position
# Refactor the whole code
# Do for the little squares, is that more robust?

def calculate_orientation(points):
    # points is a 4x2 array with the coordinates of the bounding box corners
    # Calculate the vectors between the points
    vector_01 = points[1] - points[0]
    vector_12 = points[2] - points[1]
    vector_23 = points[3] - points[2]
    vector_30 = points[0] - points[3]

    # Calculate the angles of these vectors
    angle_01 = np.arctan2(vector_01[1], vector_01[0]) * 180 / np.pi
    angle_12 = np.arctan2(vector_12[1], vector_12[0]) * 180 / np.pi
    angle_23 = np.arctan2(vector_23[1], vector_23[0]) * 180 / np.pi
    angle_30 = np.arctan2(vector_30[1], vector_30[0]) * 180 / np.pi

    # Average the angles
    average_angle = (angle_01 + angle_12 + angle_23 + angle_30) / 4

    return average_angle

CAP_WIDTH, CAP_HEIGHT = 1920, 1080#1280, 720
cap = cv2.VideoCapture(3)# 2 for droidcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)


np.set_printoptions(precision=2)


# Initialize the QRCodeDetector
detector = cv2.QRCodeDetector()

size = 5.0
params_qr = [17.54, 17.54] #shape in cm
#q1 = [[0,               0,              0],
#      [params_qr[0],    0,              0],
#      [0,               params_qr[0],   0],
#      [params_qr[0],    params_qr[1],   0]
#      ]
q1 = [[-params_qr[0]/2, params_qr[0]/2,     0],
      [params_qr[0]/2,  params_qr[0]/2,     0],
      [params_qr[0]/2,  -params_qr[0]/2,    0],
      [-params_qr[0]/2, -params_qr[0]/2,    0]
      ]
object_points  = np.array(q1)
object_points = np.expand_dims(object_points, axis=-1)

#fx = 7.18856e+02
#fy = 7.188560e+02
#cx = 6.071928e+02
#cy = 1.852157e+02
#K = np.array([[fx, 0, cx],
#              [0, fy, cy],
#              [0, 0, 1]], dtype=np.float32)

K = np.array([[2.19084549e+03, 0.00000000e+00, 9.59329012e+02],
              [0.00000000e+00, 2.19210691e+03, 5.77106533e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist_coeffs = np.array([[ 0.00169454, -0.01182477,  0.00036902, -0.00025134,  0.06841443]])#np.zeros((4, 1))  # Assuming no lens distortion


while cap.isOpened():

    success, image = cap.read()
    start = time.perf_counter()

    #value, points, qrcode = detector.detectAndDecode(img)
    # Detect and decode the QR code
    decoded_text, points, qrcode = detector.detectAndDecode(image)

    if points is not None:
        # Convert points to an array of integers
        points = points[0].astype(np.int32)
        POINTS_EXAMPLE = points

        ############################################################################
        # Convert detected points to the correct format
        image_points = np.array(points, dtype=np.float32).reshape(-1, 2)
        #image_points = np.hstack((image_points, np.ones([image_points.shape[0], 1], image_points.dtype)))
        image_points = np.expand_dims(image_points, axis=-1)
        # Calculate the orientation
        #orientation = calculate_orientation(points)
        orientation = 1
        # Solve for pose
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs, flags = cv2.SOLVEPNP_IPPE_SQUARE)
        # Convert rotation vector to rotation matrix



        if rvec is not None:
            R, _ = cv2.Rodrigues(rvec)
            # Form the transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = R
            transformation_matrix[:3, 3] = tvec.T

            ############vvvvvvvvv 3D axes on top of QR code vvvvvvvvv###################################
            # Project 3D axis points to the image plane
            axis = np.float32([[20, 0, 0], [0, 20, 0], [0, 0, -20]]).reshape(-1, 3)
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist_coeffs)

            # Ensure the projected points are integers
            imgpts = np.int32(imgpts).reshape(-1, 2)

            # Draw the coordinate axes on the image (2D drawing)
            corner = tuple(points[0].ravel())
            image = cv2.line(image, corner, tuple(imgpts[0]), (255, 0, 0), 5)  # x-axis in blue
            image = cv2.line(image, corner, tuple(imgpts[1]), (0, 255, 0), 5)  # y-axis in green
            image = cv2.line(image, corner, tuple(imgpts[2]), (0, 0, 255), 5)  # z-axis in red
            ############^^^^^^^^^ 3D axes on top of QR code ^^^^^^^^^#################################



        # Print the pose
        print("Translation vector (tvec):\n", tvec)
        print("Rotation matrix (R):\n", R)
        print("Transformation matrix:\n", transformation_matrix)
        ############################################################################

        # Draw the bounding box
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Display the image with the bounding box and orientation
        cv2.rectangle(image, (0, 170), (400, 300), (255, 255, 255), -1)
        #cv2.putText(image, f'Orientation: {orientation:.2f} degrees', (30, 70),
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(image, str( [0]), (0, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, str(transformation_matrix[1]), (0, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, str(transformation_matrix[2]), (0, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        cv2.putText(image, "x \t\t y \t\t z", (0, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, str(transformation_matrix[:,-1]), (0, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        print(f'Decoded text: {decoded_text}')
        print(f'Orientation: {orientation:.2f} degrees')
        q2 = points
        #print(transform)



    end = time.perf_counter()
    totalTime = end - start
    fps = 1 / totalTime

    cv2.putText(image, f'FPS: {int(fps)}', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow('img_orientation', image)
    if qrcode is not None:
        cv2.imshow('QR_unrotated', qrcode)
        QRCODE_EXAMPLE = qrcode

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

cv2.destroyAllWindows()