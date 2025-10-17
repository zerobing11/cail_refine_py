import numpy as np
import cv2

# 读取外参数据
input_file = '/home/lz/PycharmProjects/resize_image/evaluate/6DOF.txt'
output_file = '/home/lz/PycharmProjects/resize_image/evaluate/E_test.txt'

# 内参与畸变系数（固定的部分）
intrinsics = np.array([[909.554565, 0.000000, 630.406738],
                       [0.000000, 907.064636, 371.516388],
                       [0.000000, 0.000000, 1.000000]])

distortion = np.zeros(5)  # 假设畸变系数是零

# 处理每一行外参数据
with open(input_file, 'r') as f, open(output_file, 'w') as out:
    for line in f:
        # 跳过以#开头的注释行
        if line.startswith('#'):
            continue

        values = list(map(float, line.strip().split(',')))
        rvec = np.array(values[:3])  # 旋转向量
        tvec = np.array(values[3:])  # 平移向量

        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)

        # 输出旋转矩阵
        for i in range(3):
            out.write(f"{R[i, 0]:.6f} {R[i, 1]:.6f} {R[i, 2]:.6f}\n")

        # 输出平移向量
        out.write(f"{tvec[0]:.6f} {tvec[1]:.6f} {tvec[2]:.6f}\n")

        # 输出内参与畸变系数（固定值）
        for row in intrinsics:
            out.write(" ".join([f"{x:.6f}" for x in row]) + "\n")
        # out.write("0.000000 0.000000 1.000000\n")
        out.write("0.000000 0.000000 0.000000 0.000000 0.000000\n")
        out.write("\n")

