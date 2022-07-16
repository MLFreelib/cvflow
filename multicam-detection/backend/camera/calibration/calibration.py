import numpy as np


class Calibration:
    def __init__(self, screen_points, world_points):
        self.screen_points = screen_points
        self.world_points = world_points

        x = screen_points[:, 0]
        y = screen_points[:, 1]

        X = world_points[:, 0]
        Y = world_points[:, 1]
        Z = world_points[:, 2]

        a_x = np.array([
            -X, -Y, -Z, -np.ones(6),
            np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6),
            x * X, x * Y, x * Z, x
        ])

        a_y = np.array([
            np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6),
            -X, -Y, -Z, -np.ones(6),
            y * X, y * Y, y * Z, y
        ])

        M = np.empty((2 * 6, 12))
        M[::2, :] = a_x.T
        M[1::2, :] = a_y.T

        U, S, V = np.linalg.svd(M)

        P = V[-1, :].reshape((3, 4))

        self.matrix = P

    def project_3d_to_2d(self, coord):
        a, b, c = self.matrix @ np.append(coord, np.ones(1)).T

        return np.array([a / c, b / c])

    # noinspection PyPep8Naming
    def project_2d_to_3d(self, coord, X=None, Y=None, Z=None):
        M = np.array([
            [
                self.matrix[2, 0] * coord[0] - self.matrix[0, 0],
                self.matrix[2, 1] * coord[0] - self.matrix[0, 1],
                self.matrix[2, 2] * coord[0] - self.matrix[0, 2]
            ],
            [
                self.matrix[2, 0] * coord[1] - self.matrix[1, 0],
                self.matrix[2, 1] * coord[1] - self.matrix[1, 1],
                self.matrix[2, 2] * coord[1] - self.matrix[1, 2]
            ]
        ])

        M = np.vstack([
            ["X", "Y", "Z"],
            M
        ])

        rest_coords = np.array(["X", "Y", "Z"])
        result = np.array([
            ["X", "Y", "Z"],
            [0, 0, 0]
        ])

        v = np.array([
            self.matrix[0, 3] - self.matrix[2, 3] * coord[0],
            self.matrix[1, 3] - self.matrix[2, 3] * coord[1],
        ])

        if X is not None:
            v -= M[:, np.argwhere(M[0, :] == "X").flatten()][1:, :].astype(np.float64).flatten() * X
            M = np.delete(
                M,
                np.argwhere(M[0, :] == "X").flatten(),
                axis=1
            )
            rest_coords = np.delete(
                rest_coords,
                np.argwhere(rest_coords == "X"),
                axis=0
            )
            result[1, 0] = X

        if Y is not None:
            v -= M[:, np.argwhere(M[0, :] == "Y").flatten()][1:, :].astype(np.float64).flatten() * Y
            M = np.delete(
                M,
                np.argwhere(M[0, :] == "Y").flatten(),
                axis=1
            )
            rest_coords = np.delete(
                rest_coords,
                np.argwhere(rest_coords == "Y"),
                axis=0
            )
            result[1, 1] = Y

        if Z is not None:
            v -= M[:, np.argwhere(M[0, :] == "Z").flatten()][1:, :].astype(np.float64).flatten() * Z
            M = np.delete(
                M,
                np.argwhere(M[0, :] == "Z").flatten(),
                axis=1
            )
            rest_coords = np.delete(
                rest_coords,
                np.argwhere(rest_coords == "Z"),
                axis=0
            )
            result[1, 2] = Z

        if M.shape[1] == 0:
            print("Cannot solve equation with shape 0")
            raise Exception
        elif M.shape[1] == 1:
            divide_array = v / M[1:, :].astype(np.float64).flatten()

            if divide_array[0] == divide_array[1]:
                return divide_array[0]
            else:
                print("Cannot solve equation system (various solutions)")
                raise Exception
        elif M.shape[1] == 2:
            idxs = np.in1d(result[0, :], rest_coords)
            result = result[1, :].astype(np.float64)

            M = M[1:, :].astype(np.float64)

            result[idxs] = np.linalg.solve(M, v)
        else:
            print("Cannot solve equation with shape {}".format(M.shape[1]))
            raise Exception

        return result.reshape(3)
