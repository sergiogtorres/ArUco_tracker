import cv2
import numpy as np

import time
from utils import *

CAP_WIDTH, CAP_HEIGHT = 1920, 1080
np.set_printoptions(precision=2)

# Capture settings
cap = cv2.VideoCapture(3)  # change as needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

# Marker parameters
params_qr = [15, 15]  # shape in cm, as measured IRL
q1 = [[-params_qr[0] / 2, params_qr[0] / 2, 0],
      [params_qr[0] / 2, params_qr[0] / 2, 0],
      [params_qr[0] / 2, -params_qr[0] / 2, 0],
      [-params_qr[0] / 2, -params_qr[0] / 2, 0]
      ]
#### various arrays ##########################################
transformation_matrix_o = np.eye(4)
object_points = np.array(q1)
object_points = np.expand_dims(object_points, axis=-1)

#### Camera parameters #######################################

# Camera parameters
K = np.array([[2.19084549e+03, 0.00000000e+00, 9.59329012e+02],
              [0.00000000e+00, 2.19210691e+03, 5.77106533e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

dist_coeffs = np.array([[0.00169454, -0.01182477, 0.00036902, -0.00025134, 0.06841443]], dtype=np.float32)

#### Paremeters to draw the projected 3D axes ################
axis_length = params_qr[0]  # Length of the axes (centimeters)
#### Initialize ArUco detector################################
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)  # Select ArUco dictionary
parameters = cv2.aruco.DetectorParameters()  # Create detector parameters
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

while cap.isOpened():
    success_cap, image = cap.read()
    if not success_cap:
        break

    start = time.perf_counter()

    # For QR codes
    #decoded_text, points, qrcode = detector.detectAndDecode(image)

    # Detect and decode ArUco markers
    corners, ids, _ = detector.detectMarkers(image)

    if ids is not None:
        ids = ids.reshape(-1, )
        id_min = ids.min()
        for id, corners_idx in zip(ids, corners):
            # Convert each set of points to an array of integers
            points = corners_idx.astype(np.int32)

            # Convert detected points to the appropriate format
            image_points = np.array(points, dtype=np.float32).reshape(-1, 2)
            image_points = np.expand_dims(image_points, axis=-1)

            # Find the pose of the ith set of markers
            success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs,
                                               flags=cv2.SOLVEPNP_IPPE_SQUARE)
            # Convert rotation vector to rotation matrix##

            if rvec is not None:
                R, _ = cv2.Rodrigues(rvec)
                # Form the transformation matrix
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, :3] = R
                transformation_matrix[:3, 3] = tvec.T  #

                if id == id_min:
                    transformation_matrix_o = transformation_matrix

            # use functions from utils to project info for each marker, as well as 3D axes
            image = draw_info(points, image, transformation_matrix, transformation_matrix_o, image_points)
            image = draw_3D_axes(image, axis_length, params_qr, transformation_matrix, K, dist_coeffs)

    end = time.perf_counter()
    totalTime = end - start
    fps = 1 / totalTime

    cv2.putText(image, f'FPS: {int(fps)}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow('ArUco code detection and tracking', image)
    #    if qrcode is main_ArUco.pynot None:
    #        cv2.imshow('QR_unrotated', qrcode)
    #        QRCODE_EXAMPLE = qrcode


    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()


cv2.destroyAllWindows()
