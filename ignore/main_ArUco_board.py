import cv2
import numpy as np

import time
from utils import Perspective


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

def get_relative_coords(x, y, z, params, flag="simple"):
    # Invert the transformation matrix

    transformation_matrix = params
    inv_transformation_matrix = np.linalg.inv(transformation_matrix)

    # point in the camera frame
    point_camera = np.array((x, y, z, 1), dtype=np.float32).reshape(4, 1)  # Homogeneous coordinates

    # Transform the point to the world frame
    point_world = inv_transformation_matrix @ point_camera

    # Convert back to non-homogeneous coordinates
    point_world = point_world[:3].ravel()

    return point_world


def draw_info(image, transformation_matrix, transformation_matrix_o, image_points, absolute=False):


    u, v = image_points[0]
    u = int(u[0])
    v = int(v[0])

    # Draw the bounding box
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)


    x, y, z, _ = transformation_matrix[:, -1]
    x1, y1, z1 = get_relative_coords(x, y, z, transformation_matrix_o, flag="simple")

    text_X = f"abs: x: {x:.2f}, y: {y:.2f}, z: {z:.2f}"
    text_X1 = f"rel: x1: {x1:.2f}, y1: {y1:.2f}, z1: {z1:.2f}"
    # Get the size of the text
    (text_width_X, text_height_X), baseline_X = cv2.getTextSize(text_X, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    (text_width_X1, text_height_X1), baseline_X1 = cv2.getTextSize(text_X1, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    # Draw rectangles as background for the text
    #cv2.rectangle(image, (u, v - text_height_X - baseline_X), (u + text_width_X, v + baseline_X), (255, 255, 255), -1)
    #cv2.rectangle(image, (u, v + 20 - text_height_X1 - baseline_X1), (u + text_width_X1, v + 20 + baseline_X1),
    #              (255, 255, 255), -1)

    if absolute:
        cv2.putText(image, text_X, (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 2550), 3)
        cv2.putText(image, text_X, (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
    else:
        cv2.putText(image, text_X1, (u, v+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3)
        cv2.putText(image, text_X1, (u, v+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    q2 = points



    return image

def normalize_homogeneous(vec):
    """

    :param vec: homogeneous vector (x, y, z) in camera coordinates
    :return: euclidean vector, projection of input (z/z, y/z)
    """
    return vec/vec[2]

def project_to_2D(vec, K):
    return (vec.T @ K)[:2]/vec[2]

def generate_board():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    num_markers_side = 10
    x = ([i for i in range(num_markers_side)] +
         [num_markers_side] * num_markers_side +
         [i for i in range(num_markers_side)][::-1] +
         [0] * num_markers_side)

    y = ([0] * num_markers_side +
         [i for i in range(num_markers_side)] +
         [num_markers_side] * num_markers_side +
         [i for i in range(num_markers_side)][::-1]
         )
    z = np.zeros(np.shape(y))
    # xy = ( np.array( [x, y] ) ).T
    xyz = (np.array([x, y, z])).T * 1.2
    expanded_array = xyz[:, np.newaxis, :]  # Shape (40, 1, 2)
    broadcasted_array = np.tile(expanded_array, (1, 4, 1))  # Shape (40, 4, 2)
    objPoints = np.zeros((num_markers_side * 4, 4, 3))
    objPoints += broadcasted_array
    objPoints_old = np.copy(objPoints)

    objPoints[:, 0, :] += [0, 0, 0]  # left-top point
    objPoints[:, 1, :] += [0, 1, 0]  # right-top point
    objPoints[:, 2, :] += [1, 1, 0]  # right-bottom point
    objPoints[:, 3, :] += [1, 0, 0]  # left-bottom point
    # print(np.hstack((objPoints_old[0], objPoints[0])))

    objPoints -= objPoints.min()
    objPoints[:, :, 2] -= objPoints[0, 0, 2]
    objPoints = objPoints.astype(dtype=np.float32)
    # objPoints.min()

    objPoints_tuple = np.zeros((num_markers_side * 4, 4))
    objPoints_tuple = objPoints_tuple.tolist()
    for i, row in enumerate(objPoints):
        for j, point in enumerate(row):
            objPoints_tuple[i][j] = tuple(objPoints[i, j])

    ids = np.linspace(0, np.shape(objPoints)[0] - 1, np.shape(objPoints)[0]).astype(int)
    board = cv2.aruco.Board(objPoints, dictionary, ids)
    return board

CAP_WIDTH, CAP_HEIGHT = 1920, 1080#1280, 720
cap = cv2.VideoCapture(3)# 2 for droidcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)


np.set_printoptions(precision=2)


# Initialize the QRCodeDetector
#detector = cv2.QRCodeDetector()

#size = 5.0


params_qr = [15, 15]#[17.54, 17.54] #shape in cm
q1 = [[0,               0,              0],
      [params_qr[0],    0,              0],
      [0,               params_qr[0],   0],
      [params_qr[0],    params_qr[1],   0]
      ]
q1 = [[-params_qr[0]/2, params_qr[0]/2,     0],
      [params_qr[0]/2,  params_qr[0]/2,     0],
      [params_qr[0]/2,  -params_qr[0]/2,    0],
      [-params_qr[0]/2, -params_qr[0]/2,    0]
      ]

xo, yo, zo = 0,0,0 #origin points, will be updated with position of marker with lowest id
transformation_matrix_o = np.eye(4)
object_points  = np.array(q1)
object_points = np.expand_dims(object_points, axis=-1)

# Initialize ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)  # Select ArUco dictionary
parameters = cv2.aruco.DetectorParameters()  # Create detector parameters

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


K = np.array([[2.19084549e+03, 0.00000000e+00, 9.59329012e+02],
              [0.00000000e+00, 2.19210691e+03, 5.77106533e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist_coeffs = np.array([[ 0.00169454, -0.01182477,  0.00036902, -0.00025134,  0.06841443]])#np.zeros((4, 1))  # Assuming no lens distortion

board = generate_board()
while cap.isOpened():

    success, image = cap.read()
    start = time.perf_counter()

    #value, points, qrcode = detector.detectAndDecode(img)
    # Detect and decode the QR code
    #decoded_text, points, qrcode = detector.detectAndDecode(image)

    # Detect and decode ArUco markers
    corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners, ids, rejectedImgPoints, recoveredIdxs = detector.refineDetectedMarkers(image, board, corners, ids, rejectedImgPoints, K, dist_coeffs)

    if ids is not None:
        ids = ids.reshape(-1,)
        id_min = ids.min()
    #sorted_idx = ids.argsort()
    #ids = ids[sorted_idx]
    #corners = corners[sorted_idx]
    #print (corners, ids)
    #print (rejectedImgPoints)
################I dont think this works################################
#    if ids is not None and len(ids) > 0:
#        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, K, dist_coeffs)
#
#        for i in range(len(ids)):
#            # Draw axis for each marker
#            cv2.aruco.drawAxis(image, K, dist_coeffs, rvecs[i], tvecs[i], marker_size)
#######################################################################

    #points = corners
    if corners is not None and len(corners)>0:
        retval_board, rvec_board, tvec_board = cv2.aruco.estimate

        for id, corners_idx in zip(ids, corners):
            # Convert points to an array of integers

            #getting the info for each detected code
            points = corners_idx.astype(np.int32)
            #code_text = ids[detected_idx]
            #id = ids[detected_idx]
            #POINTS_EXAMPLE = points#

            ############################################################################
            # Convert detected points to the correct format
            image_points = np.array(points, dtype=np.float32).reshape(-1, 2)
            image_points = np.expand_dims(image_points, axis=-1)

            # Solve for pose
            success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs, flags = cv2.SOLVEPNP_IPPE_SQUARE)

            # Convert rotation vector to rotation matrix##

            if rvec is not None:
                R, _ = cv2.Rodrigues(rvec)
                # Form the transformation matrix
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, :3] = R
                transformation_matrix[:3, 3] = tvec.T#

                if id == id_min:
                    #xo, yo, zo, _ = transformation_matrix[:, -1]
                    transformation_matrix_o = transformation_matrix
                ############vvvvvvvvv 3D axes on top of QR code vvvvvvvvv###################################
                # Project 3D axis points to the image plane
    #            axis = np.float32([[20, 0, 0], [0, 20, 0], [0, 0, -20]]).reshape(-1, 3)
    #            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist_coeffs)#

                # Ensure the projected points are integers
    #            imgpts = np.int32(imgpts).reshape(-1, 2)

                # Draw the coordinate axes on the image (2D drawing)
    #            corner = tuple(points[0].ravel())
    #            image = cv2.line(image, corner, tuple(imgpts[0]), (255, 0, 0), 5)  # x-axis in blue
    #            image = cv2.line(image, corner, tuple(imgpts[1]), (0, 255, 0), 5)  # y-axis in green
    #            image = cv2.line(image, corner, tuple(imgpts[2]), (0, 0, 255), 5)  # z-axis in red
    #            ############^^^^^^^^^ 3D axes on top of QR code ^^^^^^^^^#################################


            image = draw_info(image, transformation_matrix, transformation_matrix_o, image_points)

###########################################################################################################################
            #DRAWING OF 3D AXES
            axis_length = 50.0  # Length of the axes in arbitrary units (e.g., centimeters)
            axes_points = np.float32([[0, 0, 0],
                                      [axis_length, 0, 0],
                                      [0, axis_length, 0],
                                      [0, 0, axis_length]])  # Points for X, Y, Z axes respectively

            #correction to plot on the top left corner
            axes_points -= np.array([params_qr[0]/2, params_qr[0]/2,     0])
            # Transform axes points to camera frame
            #axes_points_camera = cv2.perspectiveTransform(axes_points.reshape(-1, 1, 3), transformation_matrix_o)

            #axis = np.float32([[0, 0, 0], [20, 0, 0], [0, 20, 0], [0, 0, -20]]).reshape(-1, 3)
            rvec_o, tvec_o = transformation_matrix[:3, :3], (transformation_matrix[:3, 3]).T
            imgpts, _ = cv2.projectPoints(axes_points, rvec_o, tvec_o, K, dist_coeffs)
            origin  = imgpts[0][0].astype(int)
            x_end   = imgpts[1][0].astype(int)
            y_end   = imgpts[2][0].astype(int)
            z_end   = imgpts[3][0].astype(int)
            # Draw the axes on the video feed
            #origin  = tuple( project_to_2D( normalize_homogeneous( axes_points_camera[0].ravel() ), K ).astype(int) )
            #x_end   = tuple( project_to_2D( normalize_homogeneous( axes_points_camera[1].ravel() ), K ).astype(int) )
            #y_end   = tuple( project_to_2D( normalize_homogeneous( axes_points_camera[2].ravel() ), K ).astype(int) )
            #z_end   = tuple( project_to_2D( normalize_homogeneous( axes_points_camera[3].ravel() ), K ).astype(int) )
            print ("origin: ", origin)
            print ("x_end: ", x_end)
            print ("y_end: ", y_end)
            print ("z_end: ", z_end)
            # Draw axes lines
            image = cv2.line(image, origin, x_end, (255, 0, 0), 2)  # X-axis (red)
            image = cv2.line(image, origin, y_end, (0, 255, 0), 2)  # Y-axis (green)
            image = cv2.line(image, origin, z_end, (0, 0, 255), 2)  # Z-axis (blue)
######################################################################################################################

        #print(transform)



    end = time.perf_counter()
    totalTime = end - start
    fps = 1 / totalTime

    cv2.putText(image, f'FPS: {int(fps)}', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow('img_orientation', image)
#    if qrcode is main_ArUco.pynot None:
#        cv2.imshow('QR_unrotated', qrcode)
#        QRCODE_EXAMPLE = qrcode

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

cv2.destroyAllWindows()