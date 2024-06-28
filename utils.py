import cv2
import numpy as np

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


def draw_info(points, image, transformation_matrix, transformation_matrix_o, image_points, absolute=False):


    u, v = image_points[0]
    u = int(u[0])
    v = int(v[0])

    # Draw the bounding box
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 255), thickness=2)


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

def draw_3D_axes(image, axis_length, params_qr, transformation_matrix, K, dist_coeffs):
    # DRAWING 3D AXES
    axes_points = np.float32([[0, 0, 0],
                              [axis_length, 0, 0],
                              [0, axis_length, 0],
                              [0, 0, axis_length]])  # Points for X, Y, Z

    # correction to plot on the top left corner
    axes_points -= np.array([params_qr[0] / 2, params_qr[0] / 2, 0])
    # Transform axes points to camera frame
    # axes_points_camera = cv2.perspectiveTransform(axes_points.reshape(-1, 1, 3), transformation_matrix_o)
    # axis = np.float32([[0, 0, 0], [20, 0, 0], [0, 20, 0], [0, 0, -20]]).reshape(-1, 3)
    rvec_o, tvec_o = transformation_matrix[:3, :3], (transformation_matrix[:3, 3]).T
    imgpts, _ = cv2.projectPoints(axes_points, rvec_o, tvec_o, K, dist_coeffs)
    origin = imgpts[0][0].astype(int)
    x_end = imgpts[1][0].astype(int)
    y_end = imgpts[2][0].astype(int)
    z_end = imgpts[3][0].astype(int)
    # Draw the axes on the video feed
    #print("origin: ", origin)
    #print("x_end: ", x_end)
    #print("y_end: ", y_end)
    #print("z_end: ", z_end)
    # Draw axes lines
    image = cv2.line(image, origin, x_end, (255, 0, 0), 2)  # X-axis (red)
    image = cv2.line(image, origin, y_end, (0, 255, 0), 2)  # Y-axis (green)
    image = cv2.line(image, origin, z_end, (0, 0, 255), 2)  # Z-axis (blue)
    return image