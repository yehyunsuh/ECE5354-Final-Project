import numpy as np
import sympy as sp
import cv2


def rotation_matrix(radian_angle, matrix_type):
    if matrix_type == 'x-axis':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(radian_angle), -np.sin(radian_angle)],
            [0, np.sin(radian_angle), np.cos(radian_angle)]
        ])
    
    elif matrix_type == 'y-axis':
        rotation_matrix = np.array([
            [np.cos(radian_angle), 0, np.sin(radian_angle)],
            [0, 1, 0],
            [-np.sin(radian_angle), 0, np.cos(radian_angle)]
        ])

    elif matrix_type == 'z-axis':
        rotation_matrix = np.array([
            [np.cos(radian_angle), -np.sin(radian_angle), 0],
            [np.sin(radian_angle), np.cos(radian_angle), 0],
            [0, 0, 1]
        ])

    return rotation_matrix


def rotate_coordinates(coordinate, translation, rotation_matrix):
    # Before roation, translate the landmarks to the origin
    coordinate_origin = coordinate - translation

    # Rotate the translated coordinates
    coordinate_origin_rotated = coordinate_origin @ rotation_matrix

    # Translate the rotated coordinates back to the original position
    coordinate_rotated = coordinate_origin_rotated + translation

    return coordinate_rotated


def project_coordinates(ratio, coordinate, n_landmarks):
    x_proj = coordinate[:, 0] * ratio
    y_proj = coordinate[:, 1] * ratio
    z_proj = np.zeros(n_landmarks)
    coordinate_projected = np.array([x_proj, y_proj, z_proj]).T

    return coordinate_projected


def calculate_ellipse(H, h, r, k, l, theta, phi, landmark_proj, rotated_landmark_proj, ellipse, ellipse_reverse):
    # Calculate least square ellipse from the projection using cv2 fitellipse
    ellipse = cv2.fitEllipse(np.array(landmark_proj[:, :2], dtype=np.float32))
    (ellipse_x, ellipse_y), (minor, major), angle = ellipse

    ellipse_reverse = cv2.fitEllipse(np.array(rotated_landmark_proj[:, :2], dtype=np.float32))
    (ellipse_x_reverse, ellipse_y_reverse), (minor_reverse, major_reverse), angle_reverse = ellipse_reverse

    # Reparametrize the ellipse's axis values
    a, b = major / 2, minor / 2

    # Create an equation for ellipse using sympy
    x, y = sp.symbols('x y')

    # Ellipse equation rotated by theta, updating a and b
    ellipse_eq = \
        ((x-ellipse_x_reverse)*sp.cos(np.radians(angle_reverse+90)) + (y-ellipse_y_reverse)*sp.sin(np.radians(angle_reverse+90)))**2 / a**2 + \
        ((x-ellipse_x_reverse)*sp.sin(np.radians(angle_reverse+90)) - (y-ellipse_y_reverse)*sp.cos(np.radians(angle_reverse+90)))**2 / b**2 - 1

    # Substitute x = h
    substituted_eq = ellipse_eq.subs(x, ellipse_x_reverse)

    # Solve the equation for y
    y_solutions = sp.solve(substituted_eq, y)

    # Calculation of L 
    L = abs(float(y_solutions[0]) - float(y_solutions[1]))

    # Calculation of theta_ellipse
    left = (H-h)**2 / r**2
    right = 4*H**2 / L**2
    left_sub_right = 0 if abs(left - right) < 1e-4 else left - right

    theta_ellipse = np.degrees(np.arcsin(np.sqrt(left_sub_right)))

    print(f'k: {k}, l: {l}, Theta: {theta}, Phi: {phi}, Calculated Theta: {theta_ellipse}')
    print(f'Center: ({ellipse_x:.2f}, {ellipse_y:.2f}), Major: {a:.2f}, Minor: {b:.2f}, Angle: {angle:.2f}, L: {L:.2f}')

    return ellipse, ellipse_reverse, y_solutions, theta_ellipse