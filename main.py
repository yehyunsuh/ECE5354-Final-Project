import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import argparse
import math
import cv2
import os

from tqdm import tqdm
from angle import angle_by_gradient_difference, liaw_et_al, expectation_maximization


def main(args):
    print(f"===== k: {args.k}, l: {args.l} =====")

    # Fix the seed for reproducibility
    np.random.seed(2024)

    # Set parameters which are known in advance
    H = args.cam_d + args.proj_d
    h = args.proj_d
    r = args.r
    k = args.k
    l = args.l

    # Set parameters related to landmark prediction
    n_landmarks = 8  # number of landmarks

    # Create landmarks on the object
    theta_object = np.linspace(0, 2 * np.pi, n_landmarks)
    x_object = r * np.cos(theta_object)
    y_object = r * np.sin(theta_object)
    z_object = np.zeros(n_landmarks) + h
    landmark = np.array([x_object, y_object, z_object]).T

    # Create circle 
    theta_circle = np.linspace(0, 2 * np.pi, 1000)
    x_circle = r * np.cos(theta_circle)
    y_circle = r * np.sin(theta_circle)
    z_circle = np.zeros(1000) + h
    circle = np.array([x_circle, y_circle, z_circle]).T

    theta_array = np.linspace(0, 90, 90, dtype=int, endpoint=False)
    theta_array = np.linspace(0, 90, 9, dtype=int, endpoint=False)
    phi_array = np.linspace(0, 0, 1, dtype=int)
    # phi_array = np.linspace(0, 360, 9, dtype=int, endpoint=False)

    for phi in phi_array:
        os.makedirs(f'visualization_phi_{phi}', exist_ok=True)
        
        calculated_theta, liaw_theta = [], []
        for theta in theta_array:
            # Based on theta, rotate the landmarks on y-axis
            theta_rad = np.deg2rad(theta)
            rotation_matrix_y = np.array([
                [np.cos(theta_rad), 0, np.sin(theta_rad)],
                [0, 1, 0],
                [-np.sin(theta_rad), 0, np.cos(theta_rad)]
            ])

            # Before roation, translate the landmarks to the origin
            landmark_origin = landmark - np.array([0, 0, h])
            landmark_theta = landmark_origin @ rotation_matrix_y
            landmark_theta = landmark_theta + np.array([0, 0, h])

            circle_origin = circle - np.array([0, 0, h])
            circle_theta = circle_origin @ rotation_matrix_y
            circle_theta = circle_theta + np.array([0, 0, h])

            # Based on phi, rotate the landmarks on z-axis
            phi_rad = np.deg2rad(phi)
            rotation_matrix_z = np.array([
                [np.cos(phi_rad), -np.sin(phi_rad), 0],
                [np.sin(phi_rad), np.cos(phi_rad), 0],
                [0, 0, 1]
            ])

            # Before rotation, translate the landmarks to the origin
            landmark_theta_origin = landmark_theta - np.array([0, 0, h])
            landmark_theta_phi = landmark_theta_origin @ rotation_matrix_z
            landmark_theta_phi = landmark_theta_phi + np.array([0, 0, h])

            circle_theta_origin = circle_theta - np.array([0, 0, h])
            circle_theta_phi = circle_theta_origin @ rotation_matrix_z
            circle_theta_phi = circle_theta_phi + np.array([0, 0, h])

            # Translate the landmarks
            landmark_theta_phi = landmark_theta_phi + np.array([k, l, 0])
            circle_theta_phi = circle_theta_phi + np.array([k, l, 0])

            # Project the landmarks
            ratio = H / (H - landmark_theta_phi[:, 2])
            x_proj = landmark_theta_phi[:, 0] * ratio
            y_proj = landmark_theta_phi[:, 1] * ratio
            z_proj = np.zeros(n_landmarks)
            landmark_proj = np.array([x_proj, y_proj, z_proj]).T

            """
            This code exists just to confirm that rotation of the ellipse by -phi 
            give the same result as when phi = 0
            """
            # Based on -phi, rotate the landmarks on z-axis
            rotation_matrix_z_reverse = np.array([
                [np.cos(-phi_rad), -np.sin(-phi_rad), 0],
                [np.sin(-phi_rad), np.cos(-phi_rad), 0],
                [0, 0, 1]
            ])
            rotated_landmark_proj = landmark_proj @ rotation_matrix_z_reverse

            # Calculate least square ellipse from the projection using cv2 fitellipse
            ellipse = cv2.fitEllipse(np.array(landmark_proj[:, :2], dtype=np.float32))
            (ellipse_x, ellipse_y), (minor, major), angle = ellipse

            ellipse_reverse = cv2.fitEllipse(np.array(rotated_landmark_proj[:, :2], dtype=np.float32))
            (ellipse_x_reverse, ellipse_y_reverse), (minor_reverse, major_reverse), angle_reverse = ellipse_reverse

            # Reparametrize the ellipse
            a, b = major / 2, minor / 2

            # Create an equation for ellipse using sympy
            x, y = sp.symbols('x y')

            # Ellipse equation rotated by theta, updating a and b
            ellipse_eq = \
                ((x-ellipse_x_reverse)*sp.cos(np.radians(angle+90)) + (y-ellipse_y_reverse)*sp.sin(np.radians(angle_reverse+90)))**2 / a**2 + \
                ((x-ellipse_x_reverse)*sp.sin(np.radians(angle+90)) - (y-ellipse_y_reverse)*sp.cos(np.radians(angle_reverse+90)))**2 / b**2 - 1

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
            calculated_theta.append(theta_ellipse)

            print(f'k: {k}, l: {l}, Theta: {theta}, Phi: {phi}, Calculated Theta: {theta_ellipse}')
            print(f'Center: ({ellipse_x:.2f}, {ellipse_y:.2f}), Major: {a:.2f}, Minor: {b:.2f}, Angle: {angle:.2f}, L: {L:.2f}')

            # Perform Liaw et al. 
            liaw_angle, vertex_point_1, vertex_point_2, co_vertex_point_1, co_vertex_point_2 = liaw_et_al(
                ellipse_x_reverse, ellipse_y_reverse, a, b, angle
            )
            liaw_theta.append(liaw_angle)
            # print(f'Liaw Angle: {90-np.degrees(liaw_angle):.2f}')

            # Perform Expectation Maximization Algorithm
            # https://ieeexplore.ieee.org/document/543975
            # expectation_maximization()

            if args.figure:
                # Visualize it in 3D space
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')

                # Plot the camera
                ax.scatter(0, 0, H, c='g', marker='o', label='Camera')

                # Plot the object
                ax.scatter(
                    landmark_theta_phi[:, 0], landmark_theta_phi[:, 1], landmark_theta_phi[:, 2], 
                    c='r', marker='o', label='Object'
                )

                # Plot circle
                ax.plot(
                    circle_theta_phi[:, 0], circle_theta_phi[:, 1], circle_theta_phi[:, 2], 
                    color='black'
                )

                # Plot the projection
                ax.scatter(
                    landmark_proj[:, 0], landmark_proj[:, 1], landmark_proj[:, 2], 
                    c='b', marker='x', label='Projection'
                )

                # Plot the line between the camera, points and their projection
                for i in range(n_landmarks):
                    ax.plot(
                        [0, landmark_theta_phi[i, 0]], [0, landmark_theta_phi[i, 1]], [H, landmark_theta_phi[i, 2]], 
                        color='black', linewidth=0.5
                    )
                    ax.plot(
                        [landmark_theta_phi[i, 0], landmark_proj[i, 0]], [landmark_theta_phi[i, 1], landmark_proj[i, 1]], [landmark_theta_phi[i, 2], 0], 
                        color='black', linewidth=0.5
                    )   

                # Plot an ellipse
                rotation_matrix_ellipse = np.array([
                    [np.cos(np.radians(angle+90)), -np.sin(np.radians(angle+90))],
                    [np.sin(np.radians(angle+90)), np.cos(np.radians(angle+90))]
                ])

                t = np.linspace(0, 2 * np.pi, 1000)
                ellipse_point_x = ellipse_x + a * np.cos(t)
                ellipse_point_y = ellipse_y + b * np.sin(t)

                ellipse_point_rotated = rotation_matrix_ellipse @ np.vstack(
                    (ellipse_point_x - ellipse_x, ellipse_point_y - ellipse_y)
                )
                x_ellipse_rotated = ellipse_point_rotated[0, :] + ellipse_x
                y_ellipse_rotated = ellipse_point_rotated[1, :] + ellipse_y
                z_ellipse_rotated = np.zeros(y_ellipse_rotated.shape)

                # Plot an ellipse without Loss of Generation
                rotation_matrix_ellipse_wLoG = np.array([
                    [np.cos(np.radians(angle_reverse+90)), -np.sin(np.radians(angle_reverse+90))],
                    [np.sin(np.radians(angle_reverse+90)), np.cos(np.radians(angle_reverse+90))]
                ])

                ellipse_point_x_wLoG = ellipse_x_reverse + a * np.cos(t)
                ellipse_point_y_wLoG = ellipse_y_reverse + b * np.sin(t)

                ellipse_point_rotated_wLoG = rotation_matrix_ellipse_wLoG @ np.vstack(
                    (ellipse_point_x_wLoG - ellipse_x_reverse, ellipse_point_y_wLoG - ellipse_y_reverse)
                )
                x_ellipse_rotated_wLoG = ellipse_point_rotated_wLoG[0, :] + ellipse_x_reverse
                y_ellipse_rotated_wLoG = ellipse_point_rotated_wLoG[1, :] + ellipse_y_reverse
                z_ellipse_rotated_wLoG = np.zeros(y_ellipse_rotated_wLoG.shape)

                ax.plot(
                    x_ellipse_rotated, y_ellipse_rotated, z_ellipse_rotated, 
                    color='orange', label='Ellipse'
                )
                ax.plot(
                    x_ellipse_rotated_wLoG, y_ellipse_rotated_wLoG, z_ellipse_rotated_wLoG, 
                    color='yellow', label='Ellipse w/o LoG'
                )

                # Rotate the intersection points and plot it on the ellipse
                point1 = np.array([ellipse_x_reverse, float(y_solutions[0]), 0])
                point2 = np.array([ellipse_x_reverse, float(y_solutions[1]), 0])

                intersection_points = np.vstack((point1, point2))
                intersection_points_rotated = intersection_points @ rotation_matrix_z

                # Plot the rotated intersection points
                ax.scatter(
                    intersection_points_rotated[:, 0], intersection_points_rotated[:, 1], intersection_points_rotated[:, 2], 
                    color='purple', zorder=5, marker='x', label='Intersection Points'
                )

                # Plot the line between the rotated intersection points
                ax.plot(
                    [intersection_points_rotated[0, 0], intersection_points_rotated[1, 0]], 
                    [intersection_points_rotated[0, 1], intersection_points_rotated[1, 1]], 
                    [0, 0], 
                    color='purple', linewidth=2
                )

                # Plot the intersection points
                ax.scatter(
                    [ellipse_x_reverse, ellipse_x_reverse], [float(y_solutions[0]), float(y_solutions[1])], [0, 0], 
                    color='purple', zorder=5, label='Intersection Points w/o LoG'
                )

                # Plot the line between the intersection points
                ax.plot(
                    [ellipse_x_reverse, ellipse_x_reverse], [float(y_solutions[0]), float(y_solutions[1])], [0, 0], 
                    color='purple', linewidth=2
                )

                # Plot the vertex and co-vertex points in black
                ax.scatter(
                    vertex_point_1[0], vertex_point_1[1], 0,
                    color='black', marker='o', label='Vertex/Co-vertex Points'
                )
                ax.scatter(
                    vertex_point_2[0], vertex_point_2[1], 0,
                    color='black', marker='o'
                )
                ax.scatter(
                    co_vertex_point_1[0], co_vertex_point_1[1], 0,
                    color='black', marker='o'
                )
                ax.scatter(
                    co_vertex_point_2[0], co_vertex_point_2[1], 0,
                    color='black', marker='o'
                )

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim(-int(args.xy_plane_size/2), int(args.xy_plane_size/2))
                ax.set_ylim(-int(args.xy_plane_size/2), int(args.xy_plane_size/2))
                ax.set_zlim(0, args.cam_d + args.proj_d)
                ax.set_title(f'Theta: {theta}, Phi: {phi}, Calculated Theta: {theta_ellipse:.2f}')
                ax.legend()

                # Save the figure
                plt.savefig(
                    f'visualization_phi_{phi}/k_{k}_l_{l}_theta_{theta}_phi_{phi}.png', 
                    bbox_inches='tight', pad_inches=0
                )
                plt.close()
                print()
            else:
                print()

        if args.graph:
            # Draw a qq plot to compare the calculated theta and the actual theta
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(theta_array, calculated_theta, color='blue', label='Calculated Theta', s=5)
            ax.plot([0, 89], [0, 89], color='red', label='Ground Truth Theta')
            ax.legend()
            ax.set_xlim(-1, 91)
            ax.set_ylim(-1, 91)
            ax.set_xlabel('Ground Truth Theta')
            ax.set_ylabel('Calculated Theta')
            ax.set_title(f'QQ Plot when k = {k}, l = {l}')
            ax.set_aspect('equal', adjustable='datalim')
            # plt.show()
            plt.savefig(f'Figure/qq_plot_phi_{phi}_k_{k}_l_{l}.png', bbox_inches='tight')

            # Draw a qq plot to compare the calculated theta and the actual theta and Liaw's theta
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(theta_array, calculated_theta, color='blue', label='Proposed', s=5)
            ax.scatter(theta_array, liaw_theta, color='green', label='Liaw et al.', s=5)
            ax.plot([0, 89], [0, 89], color='red', label='Ground Truth Theta')
            ax.legend()
            ax.set_xlim(-1, 91)
            ax.set_ylim(-1, 91)
            ax.set_xlabel('Ground Truth Theta')
            ax.set_ylabel('Calculated Theta')
            ax.set_title(f'QQ Plot when k = {k}, l = {l}')
            ax.set_aspect('equal', adjustable='datalim')
            # plt.show()
            plt.savefig(f'Figure/qq_plot_phi_{phi}_k_{k}_l_{l}_with_liaw.png', bbox_inches='tight')



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