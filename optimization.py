import numpy as np

from scipy.optimize import least_squares


def objective_function():
    pass


def optimize_theta_phi():
    bounds = [(-90, 90), (-180, 180)]
    initial_guess = [np.random.uniform(-90, 90), np.random.uniform(-180, 180)]  # theta, phi

    result = least_squares(
        objective_function, 
        initial_guess,
        args = (
            ellipse_gt, 
        )
        bounds=bounds
    )


if __name__ == "__main__":
    optimize_theta_phi()