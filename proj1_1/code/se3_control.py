import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE
        self.kd = np.diag([4, 4, 4])
        self.kp = np.diag([8.3, 8.3, 8.3])
        self.kr = np.diag([300, 300, 300])
        self.kw = np.diag([30, 30, 30])

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE
        # Position controller
        r = state["x"].reshape(3,1)
        r_dot = state["v"].reshape(3,1)
        r_T = flat_output["x"].reshape(3,1)
        r_dot_T = flat_output["x_dot"].reshape(3,1)
        r_ddot_T = flat_output["x_ddot"].reshape(3,1)
        r_ddot_des = r_ddot_T - self.kd @ (r_dot - r_dot_T) - self.kp @ (r - r_T)
        F_des = self.mass * r_ddot_des + np.array([0, 0, self.mass * self.g]).reshape(3,1)
        R = Rotation.from_quat(state["q"]).as_matrix()
        b3 = R @ np.array([0, 0, 1]).reshape(3,1)
        u1 = float(b3.T @ F_des)
        if u1 < 1e-6:
            u1 = 1e-6

        # Attitude controller
        b3_des = F_des / np.linalg.norm(F_des, keepdims=True)
        a_psi = np.array([np.cos(flat_output["yaw"]), np.sin(flat_output["yaw"]), 0]).reshape(3,1)
        b2_des = np.cross(b3_des.flatten(), a_psi.flatten()).reshape(3,1) / np.linalg.norm(
            np.cross(b3_des.flatten(), a_psi.flatten()), keepdims=True)
        b1_des = np.cross(b2_des.flatten(), b3_des.flatten()).reshape(3,1)
        R_des = np.concatenate((b1_des, b2_des, b3_des), axis = 1)
        error = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = np.array([[error[2, 1]], [-error[2, 0]], [error[1, 0]]])
        # Set w_des to 0
        e_w = state["w"].reshape(3,1)
        u2 = self.inertia @ (-self.kr @ e_R - self.kw @ e_w)

        # Control input
        l = self.arm_length
        gamma = self.k_drag / self.k_thrust
        coefficient = np.array([[1, 1, 1, 1], [0, l, 0, -l], [-l, 0, l, 0], [gamma, -gamma, gamma, -gamma]])
        u = np.concatenate((np.array([[u1]]), u2), axis = 0)
        F = np.linalg.solve(coefficient, u)
        F = np.maximum(F, 0)
        w = np.sqrt(F / self.k_thrust)
        cmd_motor_speeds = np.clip(w, self.rotor_speed_min, self.rotor_speed_max)
        cmd_thrust = np.sum(F)
        cmd_moment = u2
        cmd_q = Rotation.from_matrix(R_des).as_quat()

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input