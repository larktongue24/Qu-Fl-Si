import numpy as np

class WaypointTraj(object):
    """

    """
    def __init__(self, points):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission. For a waypoint
        trajectory, the input argument is an array of 3D destination
        coordinates. You are free to choose the times of arrival and the path
        taken between the points in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Inputs:
            points, (N, 3) array of N waypoint coordinates in 3D
        """

        # STUDENT CODE HERE
        self._points = points
        if len(self._points) == 1:
            self._target = self._points[0]
            self._duration = 0.0
        else:
            self._v = 2.3
            l = np.diff(self._points, axis=0)
            d = np.linalg.norm(l, axis=1, keepdims=True)
            self._l_unit = l / d
            t = (d / self._v).flatten()
            self._t_start = np.insert(np.cumsum(t), 0, 0)
            self._duration = self._t_start[-1]



    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        if len(self._points) == 1:
            x = self._target.copy()
        else:
            if t > self._duration:
                x = self._points[-1]
            else:
                i_curr = np.searchsorted(self._t_start, t) - 1 if t > 0 else 0
                p_i = self._points[i_curr]
                l_i = self._l_unit[i_curr]
                x = p_i + self._v * l_i * (t - self._t_start[i_curr])
                x_dot = self._v * l_i

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output

if __name__ == "__main__":
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 2, 0],
        [1, 2, 3]])
    # l = np.diff(points, axis=0)
    # d = np.linalg.norm(l, axis=1, keepdims=True)
    # l_unit = l / d
    # v = 1
    # t = (d / v).flatten()
    # t_start = np.insert(np.cumsum(t), 0, 0)
    #
    # print(t_start)
    # t_curr = 0
    # i_curr = np.searchsorted(t_start, t_curr) - 1
    # print(i_curr)
    # p_i = points[i_curr]
    # l_i = l_unit[i_curr]
    # r_T = p_i + v * l_i * (t_curr - t_start[i_curr])
    # r_dot_T = v * l_i
    # print(r_T)

    # my_traj = WaypointTraj(points)
    # print(my_traj.update(3.4))
    from scipy.spatial.transform import Rotation
    roll, pitch, yaw = 0, 0, 0
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    r = Rotation.from_euler('xyz', [roll, pitch, yaw])
    q = r.as_quat()
    print(q)