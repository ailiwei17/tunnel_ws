from scipy.spatial.transform import Rotation as R
import numpy as np

if __name__ == '__main__':
    q_list = input("输入四元数:").split(",")
    q_w = float(q_list[0])
    q_x = float(q_list[1])
    q_y = float(q_list[2])
    q_z = float(q_list[3])
    t_list = input("输入平移矩阵:").split()
    x = float(t_list[0])
    y = float(t_list[1])
    z = float(t_list[2])

    obj_list = input("输入相机坐标:").split(',')
    obj_x = float(obj_list[0])
    obj_y = float(obj_list[1])
    obj_z = float(obj_list[2])

    # base_link -> camera
    m4 = np.identity(4)
    Rq = [q_x, q_y, q_z, q_w]
    Rm = R.from_quat(Rq)
    rotation_matrix = Rm.as_matrix()
    m4[0:3, 0:3] = rotation_matrix
    m4[0:3, 3] = [x, y, z]

    m4 = np.linalg.inv(m4)

    obj = [obj_x, obj_y, obj_z, 1]
    print(np.dot(m4, obj))







