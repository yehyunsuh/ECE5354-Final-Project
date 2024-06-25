import matplotlib.pyplot as plt
import numpy as np


def visualize_figure(args, H, n_landmarks, landmark_theta_phi, circle_theta_phi, landmark_proj, ellipse, ellipse_reverse, theta_ellipse, y_solutions, vertex_point, co_vertex_point, theta, phi, k, l, rotation_matrix_z):
    (ellipse_x, ellipse_y), (minor, major), angle = ellipse
    (ellipse_x_reverse, ellipse_y_reverse), (minor_reverse, major_reverse), angle_reverse = ellipse_reverse
    
    vertex_point_1, vertex_point_2 = vertex_point
    co_vertex_point_1, co_vertex_point_2 = co_vertex_point

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
    ellipse_point_x = ellipse_x + (major / 2) * np.cos(t)
    ellipse_point_y = ellipse_y + (minor / 2) * np.sin(t)

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

    ellipse_point_x_wLoG = ellipse_x_reverse + (major / 2) * np.cos(t)
    ellipse_point_y_wLoG = ellipse_y_reverse + (minor / 2) * np.sin(t)

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
        f'visualization/phi_{phi}/k_{k}_l_{l}_theta_{theta}_phi_{phi}.png', 
        bbox_inches='tight', pad_inches=0
    )
    plt.close()


def visualize_graph(theta_array, calculated_theta, liaw_theta, k, l, phi):
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