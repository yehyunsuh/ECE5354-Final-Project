import numpy as np

from scipy.optimize import least_squares


def objective_function():
    pass


def optimize_theta_phi():
    bounds = [(-180, 180), (-180, 180)]
    initial_guess = [np.random.uniform(-180, 180), np.random.uniform(-180, 180)]

    result = least_squares(
        objective_function, 
        initial_guess,
        bounds=bounds
    )


if __name__ == "__main__":
    optimize_theta_phi()