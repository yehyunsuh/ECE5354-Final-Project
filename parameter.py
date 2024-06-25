import numpy as np


def parameter_known(args):
    # Set parameters which are known in advance
    H = args.cam_d + args.proj_d
    h = args.proj_d
    r = args.r
    k = args.k
    l = args.l

    return H, h, r, k, l


def parameter_landmark(args, r, h):
    # Set parameters related to landmark prediction
    n_landmarks = 8  # number of landmarks

    # Create landmarks on the object
    theta_object = np.linspace(0, 2 * np.pi, n_landmarks)
    x_object = r * np.cos(theta_object)
    y_object = r * np.sin(theta_object)
    z_object = np.zeros(n_landmarks) + h
    landmark = np.array([x_object, y_object, z_object]).T

    return n_landmarks, landmark


def parameter_circle(args, r, h):
    # Create circle 
    theta_circle = np.linspace(0, 2 * np.pi, 1000)
    x_circle = r * np.cos(theta_circle)
    y_circle = r * np.sin(theta_circle)
    z_circle = np.zeros(1000) + h
    circle = np.array([x_circle, y_circle, z_circle]).T

    return circle


def parameter_rotation(args):
    theta_array = np.linspace(0, 90, 90, dtype=int, endpoint=False)
    phi_array = np.linspace(0, 360, 9, dtype=int, endpoint=False)

    return theta_array, phi_array