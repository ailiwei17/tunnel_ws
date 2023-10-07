import numpy as np


def angles_rad(angles):
    return np.array([angles[0] * 3.1415/180, angles[1] * 3.1415/180, angles[2] * 3.1415/180])


if __name__ == '__main__':
    while (1):
        zyx_list = input("输入:").split(",")
        rx = float(zyx_list[0])
        ry = float(zyx_list[1])
        rz = float(zyx_list[2])
        print(angles_rad([rx, ry, rz]))
