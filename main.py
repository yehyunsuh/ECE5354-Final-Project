import numpy as np
import argparse
import os

from tqdm import tqdm

from angle import angle_by_gradient_difference, liaw_et_al, expectation_maximization
from parameter import parameter_known, parameter_landmark, parameter_circle, parameter_rotation
from utils import rotation_matrix, rotate_coordinates, project_coordinates, calculate_ellipse
from visualization import visualize_figure, visualize_graph


def main(args):
    print(f"===== k: {args.k}, l: {args.l} =====")

    # Fix the seed for reproducibility
    np.random.seed(2024)

    H, h, r, k, l = parameter_known(args)
    n_landmarks, landmark = parameter_landmark(args, r, h)
    circle = parameter_circle(args, r, h)
    theta_array, phi_array = parameter_rotation(args)

    for phi in phi_array:
        os.makedirs(f'visualization/phi_{phi}', exist_ok=True)
        
        calculated_theta, liaw_theta = [], []
        for theta in theta_array:
            # Based on theta, rotate the landmarks on y-axis
            rotation_matrix_y = rotation_matrix(np.deg2rad(theta), 'y-axis')
            landmark_theta = rotate_coordinates(landmark, np.array([0, 0, h]), rotation_matrix_y)
            circle_theta = rotate_coordinates(circle, np.array([0, 0, h]), rotation_matrix_y)

            # Perform Expectation Maximization Algorithm
            # https://ieeexplore.ieee.org/document/543975
            # TODO: Guess phi
            # TODO: Calculate theta based on guessed phi
            # TODO: Take a derivate of the error of the parameters with respect to phi (ellipse parameters or distances)
            # TODO: Optimize phi using the derivative

            # How about using optimization method from scipy.optimize.least_squres to optimize theta and phi
            # The goal would be to minimize the error of the parameters for the ellipse
            # The parameters are the center of the ellipse, the major and minor axis, and the angle of the ellipse
            # The error would be the difference between the calculated ellipse and the least square ellipse
            

            # Based on phi, rotate the landmarks on z-axis
            rotation_matrix_z = rotation_matrix(np.deg2rad(phi), 'z-axis')
            landmark_theta_phi = rotate_coordinates(landmark_theta, np.array([0, 0, h]), rotation_matrix_z)
            circle_theta_phi = rotate_coordinates(circle_theta, np.array([0, 0, h]), rotation_matrix_z)

            # Translate the landmarks
            landmark_theta_phi = landmark_theta_phi + np.array([k, l, 0])
            circle_theta_phi = circle_theta_phi + np.array([k, l, 0])

            # Project the landmarks
            ratio = H / (H - landmark_theta_phi[:, 2])
            landmark_proj = project_coordinates(ratio, landmark_theta_phi, n_landmarks)

            """
            This code exists just to confirm that rotation of the ellipse by -phi 
            give the same result as when phi = 0
            """
            # Based on -phi, rotate the landmarks on z-axis
            rotation_matrix_z_reverse = rotation_matrix(-np.deg2rad(phi), 'z-axis')
            rotated_landmark_proj = landmark_proj @ rotation_matrix_z_reverse

            ellipse, ellipse_reverse, y_solutions, theta_ellipse = calculate_ellipse(H, h, r, k, l, theta, phi, landmark_proj, rotated_landmark_proj, ellipse, ellipse_reverse)
            calculated_theta.append(theta_ellipse)

            (ellipse_x, ellipse_y), (minor, major), angle = ellipse
            (ellipse_x_reverse, ellipse_y_reverse), (minor_reverse, major_reverse), angle_reverse = ellipse_reverse

            # Perform Liaw et al. 
            liaw_angle, vertex_point, co_vertex_point = liaw_et_al(
                ellipse_x_reverse, ellipse_y_reverse, major/2, minor/2, angle_reverse
            )
            liaw_theta.append(liaw_angle)

            if args.figure:
                visualize_figure(
                    args, H, n_landmarks, 
                    landmark_theta_phi, circle_theta_phi, landmark_proj, 
                    ellipse, ellipse_reverse, y_solutions, theta_ellipse,
                    vertex_point, co_vertex_point, 
                    theta, phi, k, l, rotation_matrix_z)
                print()
            else:
                print()

        if args.graph:
            visualize_graph(theta_array, calculated_theta, liaw_theta, k, l, phi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hip Implant Orientation')

    # parameters for the c-arm
    parser.add_argument('--cam_d', '--camera_distance', type=int, default=500, help='Distance (mm) of the camera from the object')
    parser.add_argument('--proj_d', '--projection_distance', type=int, default=500, help='Distance (mm) of the projection plane from the object')

    # parameters for 3D space
    parser.add_argument('--xy_plane_size', type=int, default=400, help='Size of the xy-plane')

    # parameters for the object
    parser.add_argument('--r', '--object_radius', type=int, default=30, help='Radius (mm) of the object')
    parser.add_argument('--k', '--translation_k', type=int, default=0, help='Translation of the object in the x-axis')
    parser.add_argument('--l', '--translation_l', type=int, default=0, help='Translation of the object in the y-axis')

    # parameters for visualization
    parser.add_argument('--graph', action='store_true', help='Whether to show the graph')
    parser.add_argument('--figure', action='store_true', help='Whether to show the figure')

    args = parser.parse_args()

    main(args)