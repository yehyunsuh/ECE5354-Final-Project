import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import argparse
import cv2
import os

from tqdm import tqdm


def main(args):
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

    theta_array = np.linspace(0, 89, 90, dtype=int)
    theta_array = np.linspace(0, 10, 11, dtype=int)
    phi_array = np.linspace(0, 0, 1, dtype=int)

    calculated_theta = []
    os.makedirs('visualization', exist_ok=True)

    # for theta in tqdm(theta_array):
    for theta in theta_array:
        for phi in phi_array:
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

            # Calculate least square ellipse from the projection using cv2 fitellipse
            ellipse = cv2.fitEllipse(np.array(landmark_proj[:, :2], dtype=np.float32))
            (ellipse_x, ellipse_y), (minor, major), angle = ellipse
            angle += 90

            # Reparametrize the ellipse
            a, b = major / 2, minor / 2

            # Create an equation for ellipse using sympy
            x, y = sp.symbols('x y')

            # Equation of the ellipse (rotated and translated)
            ellipse_eq = ((x-k)*sp.cos(angle) + (y-l)*sp.sin(angle))**2 / a**2 + ((x-k)*sp.sin(angle) - (y-l)*sp.cos(angle))**2 / b**2 - 1

            # Simplify the equation
            ellipse_eq_simplified = sp.simplify(ellipse_eq)

            # Substitute x or y into the ellipse equation
            substituted_eq_x = ellipse_eq.subs(x, k) 
            substituted_eq_y = ellipse_eq.subs(y, l)

            # Solve the equation for x or y
            x_solutions = sp.solve(substituted_eq_x, y)
            y_solutions = sp.solve(substituted_eq_y, x)

            # Calculation of L 
            # L = abs(float(y_solutions[0]) - float(y_solutions[1]))
            L = abs(float(y_solutions[0]) - float(y_solutions[1]))

            # # Calculation of theta_ellipse
            left = (H-h)**2 / r**2
            right = 4*H**2 / L**2
            right = 4*H**2 / major**2
            
            left_sub_right = 0 if abs(left - right) < 1e-4 else left - right
            theta_ellipse = np.degrees(np.arcsin(np.sqrt(left_sub_right)))
            calculated_theta.append(theta_ellipse)

            print(f'Theta: {theta}, Phi: {phi}, Calculated Theta: {theta_ellipse}')
            print(f'Ellipse: Center: ({ellipse_x}, {ellipse_y}), Major: {a}, Minor: {b}, Angle: {angle}')
            print(f'Left: {left}, Right: {right}, Left - Right: {left_sub_right}')

            # Visualize it in 3D space
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Plot the camera
            ax.scatter(0, 0, H, c='g', marker='o', label='Camera')

            # Plot the object
            ax.scatter(landmark_theta_phi[:, 0], landmark_theta_phi[:, 1], landmark_theta_phi[:, 2], c='r', marker='o', label='Object')

            # Plot the projection
            ax.scatter(landmark_proj[:, 0], landmark_proj[:, 1], landmark_proj[:, 2], c='b', marker='x', label='Projection')

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

            # Plot circle
            ax.plot(circle_theta_phi[:, 0], circle_theta_phi[:, 1], circle_theta_phi[:, 2], color='black', label='Circle')

            # Plot the ellipse
            ellipse = cv2.ellipse2Poly((int(ellipse_x), int(ellipse_y)), (int(a), int(b)), int(angle), 0, 360, 1)
            ax.plot(ellipse[:, 0], ellipse[:, 1], 0, color='orange', linewidth=2, label='Ellipse')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            ax.set_zlim(0, 1000)
            ax.set_title(f'Theta: {theta}, Phi: {phi}, Calculated Theta: {theta_ellipse:.2f}')
            ax.legend()

            # Save the figure
            plt.savefig(f'visualization/theta_{theta}_phi_{phi}.png')
            print()

    # Draw a qq plot to compare the calculated theta and the actual theta
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(theta_array, calculated_theta, color='blue', label='Calculated Theta', s=5)
    ax.plot([0, 89], [0, 89], color='red', label='Ground Truth Theta')
    ax.legend()
    ax.set_xlim(-1, 91)
    ax.set_ylim(-1, 90)
    ax.set_xlabel('Ground Truth Theta')
    ax.set_ylabel('Calculated Theta')
    ax.set_title(f'QQ Plot when k = {k}, l = {l}')
    ax.set_aspect('equal', adjustable='datalim')
    # plt.show()
    plt.savefig(f'Figure/qq_plot_k_{k}_l_{l}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hip Implant Orientation')

    # parameters for the c-arm
    parser.add_argument('--cam_d', '--camera_distance', type=int, default=500, help='Distance (mm) of the camera from the object')
    parser.add_argument('--proj_d', '--projection_distance', type=int, default=500, help='Distance (mm) of the projection plane from the object')

    # parameters for the object
    parser.add_argument('--r', '--object_radius', type=int, default=30, help='Radius (mm) of the object')
    parser.add_argument('--k', '--translation_k', type=int, default=0, help='Translation of the object in the x-axis')
    parser.add_argument('--l', '--translation_l', type=int, default=0, help='Translation of the object in the y-axis')

    args = parser.parse_args()

    main(args)