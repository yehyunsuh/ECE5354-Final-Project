import numpy as np
import sympy as sp
import math
import cv2


def angle_by_gradient_difference(x1: float, y1: float, x2: float, y2: float,
                                 x3: float, y3: float) -> list:
    theta1 = math.degrees(math.atan((y3-y2)/(x3-x2)))
    theta2 = math.degrees(math.atan((y1-y2)/(x1-x2)))
    theta = abs(theta2 - theta1)

    return theta


def liaw_et_al(ellipse_x, ellipse_y, a, b, angle_reverse):
    # Direction vectors for the major and minor axes
    major_axis_direction = np.array([np.cos(np.radians(angle_reverse+90)), np.sin(np.radians(angle_reverse+90))])
    minor_axis_direction = np.array([-np.sin(np.radians(angle_reverse+90)), np.cos(np.radians(angle_reverse+90))])  # Perpendicular to the major axis

    # Vertex points (endpoints of the major axis)
    center = np.array([ellipse_x, ellipse_y])
    vertex_point_1 = center + a * major_axis_direction
    vertex_point_2 = center - a * major_axis_direction

    # Co-vertex points (endpoints of the minor axis)
    co_vertex_point_1 = center + b * minor_axis_direction
    co_vertex_point_2 = center - b * minor_axis_direction

    angle_between_pixels = angle_by_gradient_difference(
        co_vertex_point_1[0], co_vertex_point_1[1],
        vertex_point_1[0], vertex_point_1[1], 
        vertex_point_2[0], vertex_point_2[1]
    )
    
    liaw_angle = 90 - np.degrees(np.arcsin(np.tan(np.radians(angle_between_pixels))))
    vertex_point = [vertex_point_1, vertex_point_2]
    co_vertex_point = [co_vertex_point_1, co_vertex_point_2]

    return liaw_angle, vertex_point, co_vertex_point


def expectation_maximization(landmark_proj):
    # TODO 1: Randomly pick random_phi in between 0 and 360 degrees
    random_phi = np.random.uniform(0, 360)

    # From random_phi, rotate the ellipse by -random_phi
    rotation_matrix_z_reverse = np.array([
        [np.cos(-random_phi), -np.sin(-random_phi), 0],
        [np.sin(-random_phi), np.cos(-random_phi), 0],
        [0, 0, 1]
    ])
    rotated_landmark_proj = landmark_proj @ rotation_matrix_z_reverse

    # Calculate least square ellipse from the projection
    temporary_ellipse = cv2.fitEllipse(rotated_landmark_proj)
    (temporary_ellipse_x, temporary_ellipse_y), (temporary_minor, temporary_major), temporary_angle = temporary_ellipse

    # Reparameterize the ellipse's axis values
    tmp_a, tmp_b = temporary_minor/2, temporary_major/2

    # Create an equation for ellipse using sympy
    x, y = sp.symbols('x y')

    # Ellipse equation rotated by theta, updating a and b
    ellipse_eq = \
        ((x-temporary_ellipse_x)*sp.cos(np.radians(angle+90)) + (y-temporary_ellipse_y)*sp.sin(np.radians(angle_reverse+90)))**2 / a**2 + \
        ((x-temporary_ellipse_x)*sp.sin(np.radians(angle+90)) - (y-temporary_ellipse_y)*sp.cos(np.radians(angle_reverse+90)))**2 / b**2 - 1


    # TODO 2: Based on random_phi, calculate theta


    # TODO 3: Based on theta, create an temporary_ellipse

    # TODO 4: Compare the temporary_ellipse with the original ellipse
    # Here, we calculate the distance (a.k.a loss) between the two ellipses, 
    # the angle between the two ellipses, 
    # and the major and minor axes of the temporary_ellipse

    # TODO 5: Until the temporary_ellipse is close enough to the original ellipse, repeat steps 2-4
    # Here, redefine phi based on the loss increase or decrease