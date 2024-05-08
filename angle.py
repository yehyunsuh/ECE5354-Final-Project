import numpy as np
import math


def angle_by_gradient_difference(x1: float, y1: float, x2: float, y2: float,
                                 x3: float, y3: float) -> list:
    theta1 = math.degrees(math.atan((y3-y2)/(x3-x2)))
    theta2 = math.degrees(math.atan((y1-y2)/(x1-x2)))
    theta = abs(theta2 - theta1)

    return theta


def liaw_et_al(ellipse_x, ellipse_y, a, b, angle):
    # Direction vectors for the major and minor axes
    major_axis_direction = np.array([np.cos(np.radians(angle+90)), np.sin(np.radians(angle+90))])
    minor_axis_direction = np.array([-np.sin(np.radians(angle+90)), np.cos(np.radians(angle+90))])  # Perpendicular to the major axis

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

    return liaw_angle, vertex_point_1, vertex_point_2, co_vertex_point_1, co_vertex_point_2


def expectation_maximization():
    print(1)

    # TODO 1: Randomly pick phi

    # TODO 2: Based on phi, calculate theta

    # TODO 3: Based on theta, create an temporary_ellipse

    # TODO 4: Compare the temporary_ellipse with the original ellipse
    # Here, we calculate the distance between the two ellipses, 
    # the angle between the two ellipses, 
    # and the major and minor axes of the temporary_ellipse

    # TODO 5: Until the temporary_ellipse is close enough to the original ellipse, repeat steps 1-4