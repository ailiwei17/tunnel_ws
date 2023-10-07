import numpy as np


def zyx_to_xyz_euler_angles(zyx_angles):
    """
    将ZYX型欧拉角转换为XYZ型欧拉角。
    参数：
        zyx_angles：ZYX型欧拉角，形如[phi, theta, psi]，单位为弧度。
    返回值：
        xyz_angles：XYZ型欧拉角，形如[alpha, beta, gamma]，单位为弧度。
    """
    phi, theta, psi = zyx_angles[0], zyx_angles[1], zyx_angles[2]

    R_z = np.array([[np.cos(phi), -np.sin(phi), 0],
                    [np.sin(phi), np.cos(phi), 0],
                    [0, 0, 1]])

    R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(psi), -np.sin(psi)],
                    [0, np.sin(psi), np.cos(psi)]])

    R = np.dot(np.dot(R_x, R_y), R_z)

    beta = np.arctan2(-R[2][0], np.sqrt(R[0][0] ** 2 + R[1][0] ** 2))
    alpha = np.arctan2(R[1][0] / np.cos(beta), R[0][0] / np.cos(beta))
    gamma = np.arctan2(R[2][1] / np.cos(beta), R[2][2] / np.cos(beta))

    return np.array([alpha * 180 / 3.1415, beta * 180 / 3.1415 , gamma* 180 / 3.1415 ])


if __name__ == '__main__':
    while (1):
        zyx_list = input("输入zyx:").split(",")
        rz = float(zyx_list[0])
        ry = float(zyx_list[1])
        rx = float(zyx_list[2])
        print(zyx_to_xyz_euler_angles([rz, ry, rx]))
