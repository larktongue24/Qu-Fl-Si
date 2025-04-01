# %% Imports

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


# %%

def complementary_filter_update(initial_rotation, angular_velocity, linear_acceleration, dt):
    """
    Implements a complementary filter update

    :param initial_rotation: rotation_estimate at start of update
    :param angular_velocity: angular velocity vector at start of interval in radians per second
    :param linear_acceleration: linear acceleration vector at end of interval in meters per second squared
    :param dt: duration of interval in seconds
    :return: final_rotation - rotation estimate after update
    """

    # TODO Your code here - replace the return value with one you compute
    # Gyro measurement
    R_gyro = initial_rotation * Rotation.from_rotvec(angular_velocity * dt)

    # Gravity estimation
    g = np.array([1.0, 0.0, 0.0])
    g_prime = R_gyro.apply(linear_acceleration)
    g_prime /= norm(g_prime)

    # Rotation correction
    omega_acc = np.cross(g_prime, g)
    if norm(omega_acc) < 1e-8:
        q_acc = Rotation.identity().as_quat()
    else:
        theta = np.arccos(np.clip(np.dot(g_prime, g), -1.0, 1.0))
        q_acc = Rotation.from_rotvec(omega_acc / norm(omega_acc) * theta).as_quat()

    # Adaptive gain
    e_m = abs(norm(linear_acceleration) / 9.81 - 1)
    if e_m >= 0.2:
        alpha = 0
    elif e_m <= 0.1:
        alpha = 1
    else:
        alpha = 2 - 10 * e_m

    # Blended quaternion
    q_I = np.array([0.0, 0.0, 0.0, 1.0])
    delta_q_acc = (1 - alpha) * q_I + alpha * q_acc
    delta_q_acc /= norm(delta_q_acc)

    # Apply correction
    return Rotation.from_quat(delta_q_acc) * R_gyro
