import cv2
import numpy as np

import time

one = False
mine = False
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
# Define the padding size (top, bottom, left, right)
size_pixels  = 6
side_pixels = 200
padding_pixels = 1
padding_top = int(padding_pixels * side_pixels/size_pixels)
padding_bottom = padding_top
padding_left = padding_top
padding_right = padding_top
# Add white padding
white = [255, 255, 255]  # RGB value for white color

# Save or display the image
#cv2.imwrite('padded_image.jpg', padded_image)
#cv2.imshow('Padded Image', padded_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
if one:
    for i in range(50):
        code = cv2.aruco.generateImageMarker(dictionary, i, side_pixels, 1)

        code = cv2.copyMakeBorder(code, padding_top, padding_bottom, padding_left, padding_right,
                                          cv2.BORDER_CONSTANT, value=white)
        cv2.imwrite('./aruco_DICT_6X6_250/padded/'+'code_'+str(i)+'.png', code)

else:
    if mine:
        #num_markers = 20
        #x = np.linspace(0, 19, num_markers)
        num_markers_side = 10
        x = ([i for i in range(num_markers_side)]           +
             [num_markers_side]*num_markers_side            +
             [i for i in range(num_markers_side)][::-1]     +
             [0]*num_markers_side)

        y = ([0]*num_markers_side                           +
             [i for i in range(num_markers_side)]           +
             [num_markers_side]*num_markers_side            +
             [i for i in range(num_markers_side)][::-1]
             )
        z = np.zeros(np.shape(y))
        #xy = ( np.array( [x, y] ) ).T
        xyz = ( np.array( [x, y, z] ) ).T * 1.2
        expanded_array = xyz[:, np.newaxis, :]  # Shape (40, 1, 2)
        broadcasted_array = np.tile(expanded_array, (1, 4, 1))  # Shape (40, 4, 2)
        objPoints = np.zeros((num_markers_side*4, 4, 3))
        objPoints += broadcasted_array
        objPoints_old = np.copy(objPoints)

        objPoints[:, 0, :] += [0, 0, 0]  # left-top point
        objPoints[:, 1, :] += [0, 1, 0]  # right-top point
        objPoints[:, 2, :] += [1, 1, 0]  # right-bottom point
        objPoints[:, 3, :] += [1, 0, 0 ]  # left-bottom point
        #print(np.hstack((objPoints_old[0], objPoints[0])))

        objPoints -= objPoints.min()
        objPoints[:,:,2] -= objPoints[0,0,2]
        objPoints = objPoints.astype(dtype=np.float32)
        #objPoints.min()

        objPoints_tuple = np.zeros((num_markers_side*4, 4))
        objPoints_tuple = objPoints_tuple.tolist()
        for i, row in enumerate(objPoints):
            for j, point in enumerate(row):
                objPoints_tuple[i][j] =tuple(objPoints[i,j])


        ids = np.linspace(0, np.shape(objPoints)[0]-1, np.shape(objPoints)[0]).astype(int)
        board = cv2.aruco.Board(objPoints, dictionary, ids)
        im_me = board.generateImage((1000,1000)) # [, img[, marginSize[, borderBits]]]
        cv2.imwrite('./' + 'test_board_me_2'+ '.png', im_me)
    else: # mine == False
        #this option uses cv2.aruco.GridBoard((nx, ny), markerLength, markerSeparation, aruco_dict)
        # simpler syntax, makes a fully covered board
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        markerLength = 40   # Here, our measurement unit is centimetre.
        markerSeparation = 8   # Here, our measurement unit is centimetre.

        board = cv2.aruco.GridBoard((5, 7), markerLength, markerSeparation, aruco_dict)

        arucoParams = cv2.aruco.DetectorParameters_create()



#############################################################################################
#    #testing for nor...
#    p1 = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
#    p2 = np.array([[1, 0, 0], [1, 1, 0], [2, 1, 0], [2, 0, 0]], dtype=np.float32)
#    objPoints2 = np.array([p1, p2])
#    #dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
#    ids = np.array([0, 1])
#    board = cv2.aruco.Board(objPoints2, dictionary, ids)
#    img = board.generateImage((100, 100))
#    cv2.imwrite('./' + 'test_board' + str(ids) + '.png', img)
########################################################################################