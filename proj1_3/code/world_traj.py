import numpy as np

from .graph_search import graph_search, rdp_simplify, resample


class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.2, 0.2, 0.2])
        self.margin = 0.6

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.

        # Remove collinear points
        # self.threshold = 0.1
        # self.points = simplify(self.path, self.threshold)
        # self.points = resample_path(self.path, 1.0)

        simplified = rdp_simplify(self.path, epsilon=0.5)
        self.points = resample(simplified, 1.5)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE
        # Flight time segment based on constant speed and distance between waypoints
        self.v = 3.6
        l = np.diff(self.points, axis=0)
        d = np.linalg.norm(l, axis=1, keepdims=True)
        t = (d / self.v).flatten()
        self.t_seg = t
        self.t_acc = np.insert(np.cumsum(t), 0, 0)
        self.duration = self.t_acc[-1]

        self.n_segments = self.points.shape[0] - 1
        self.coefficient = np.zeros((3 * self.n_segments, 6))

        for axis in range(3):
            A = np.zeros((6 * self.n_segments, 6 * self.n_segments))
            b = np.zeros(6 * self.n_segments)

            # Speed and acceleration constraint at the start point
            A[0, 4] = 1  # speed
            A[1, 3] = 2  # acc
            for i in range(self.n_segments):
                seg_index = 6 * i
                t = self.t_seg[i]

                # Position constraint at the start point of each segment
                A[seg_index + 2, seg_index + 5] = 1
                b[seg_index + 2] = self.points[i, axis]

                # Position constraint at the end point of each segment
                A[seg_index + 3, seg_index : seg_index + 6] = [t ** 5, t ** 4, t ** 3, t ** 2, t, 1]
                b[seg_index + 3] = self.points[i + 1, axis]

                # Speed and acceleration constraint at the end point
                if i == self.n_segments - 1:
                    A[seg_index + 4, seg_index : seg_index + 6] = [5 * t ** 4, 4 * t ** 3, 3 * t ** 2, 2 * t, 1, 0]
                    A[seg_index + 5, seg_index : seg_index + 6] = [20 * t ** 3, 12 * t ** 2, 6 * t, 2, 0, 0]
                    break

                # Continuity constraint at intermediate points
                # Speed continuity
                A[seg_index + 4, seg_index : seg_index + 6] = [5 * t ** 4, 4 * t ** 3, 3 * t ** 2, 2 * t, 1, 0]
                A[seg_index + 4, seg_index + 10] = -1
                # Acceleration continuity
                A[seg_index + 5, seg_index : seg_index + 6] = [20 * t ** 3, 12 * t ** 2, 6 * t, 2, 0, 0]
                A[seg_index + 5, seg_index + 9] = -2
                # Jerk continuity
                A[seg_index + 6, seg_index : seg_index + 6] = [60 * t ** 2, 24 * t, 6, 0, 0, 0]
                A[seg_index + 6, seg_index + 8] = -6
                # Snap continuity
                A[seg_index + 7, seg_index : seg_index + 6] = [120 * t, 24, 0, 0, 0, 0]
                A[seg_index + 7, seg_index + 7] = -24

            self.coefficient[axis * self.n_segments : (axis + 1) * self.n_segments, :] = np.linalg.solve(A, b).reshape(self.n_segments, 6)


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
        # After reaching the goal
        if t >= self.duration:
            x = self.points[-1]
        else:
            i_curr = np.searchsorted(self.t_acc, t) - 1 if t > 0 else 0
            t_start = self.t_acc[i_curr]
            dt = t - t_start

            P = np.array([dt ** 5, dt ** 4, dt ** 3, dt ** 2, dt, 1])
            V = np.array([5 * dt ** 4, 4 * dt ** 3, 3 * dt ** 2, 2 * dt, 1, 0])
            A = np.array([20 * dt ** 3, 12 * dt ** 2, 6 * dt, 2, 0, 0])
            J = np.array([60 * dt ** 2, 24 * dt, 6, 0, 0, 0])
            S = np.array([120 * dt, 24, 0, 0, 0, 0])

            for axis in range(3):
                coefficient = self.coefficient[axis * self.n_segments + i_curr, :]
                x[axis] = coefficient @ P.T
                x_dot[axis] = coefficient @ V.T
                x_ddot[axis] = coefficient @ A.T
                x_dddot[axis] = coefficient @ J.T
                x_ddddot[axis] = coefficient @ S.T

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
