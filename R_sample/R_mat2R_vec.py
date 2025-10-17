import numpy as np
from scipy.spatial.transform import Rotation
import os

input_file = '/home/lz/PycharmProjects/resize_image/R_sample/R_mat.txt'
output_file = '/home/lz/PycharmProjects/resize_image/R_sample/R_vec.txt'

try:
    with open(input_file, 'r') as f_in:
        with open(output_file, 'w') as f_out:
            lines = f_in.readlines()
            matrix_rows = []
            matrix_count = 0

            for line in lines:
                clean_line = line.strip()
                if not clean_line:
                    if len(matrix_rows) == 3:
                        try:
                            # Convert the rows into a 3x3 numpy array
                            rot_mat = np.array(matrix_rows)

                            # Create a Rotation object
                            r = Rotation.from_matrix(rot_mat)

                            # Get the rotation vector
                            rot_vec = r.as_rotvec()

                            # Calculate the angle (in radians)
                            angle_rad = np.linalg.norm(rot_vec)

                            # Handle the case of zero rotation
                            if angle_rad < 1e-6:
                                axis = np.array([0.0, 0.0, 0.0])  # Or any arbitrary axis
                                angle_deg = 0.0
                            else:
                                # Calculate the unit vector
                                axis = rot_vec / angle_rad
                                # Convert the angle to degrees
                                angle_deg = np.degrees(angle_rad)

                            # Write the unit vector and angle to the file
                            f_out.write(' '.join(map(str, axis)) + ' ' + str(angle_deg) + '\n')
                            matrix_count += 1
                        except Exception as e:
                            print(f"处理第 {matrix_count + 1} 个矩阵时出错：{e}")
                            f_out.write("转换失败\n")

                    matrix_rows = []
                    continue

                row = [float(x) for x in clean_line.split()]
                if len(row) == 3:
                    matrix_rows.append(row)
                else:
                    print(f"警告: 忽略格式不正确的行: {line.strip()}")

            if len(matrix_rows) == 3:
                try:
                    rot_mat = np.array(matrix_rows)
                    r = Rotation.from_matrix(rot_mat)
                    rot_vec = r.as_rotvec()
                    angle_rad = np.linalg.norm(rot_vec)
                    if angle_rad < 1e-6:
                        axis = np.array([0.0, 0.0, 0.0])
                        angle_deg = 0.0
                    else:
                        axis = rot_vec / angle_rad
                        angle_deg = np.degrees(angle_rad)
                    f_out.write(' '.join(map(str, axis)) + ' ' + str(angle_deg) + '\n')
                    matrix_count += 1
                except Exception as e:
                    print(f"处理第 {matrix_count + 1} 个矩阵时出错：{e}")
                    f_out.write("转换失败\n")

    print(f"已成功转换 {matrix_count} 个旋转矩阵，结果已保存到 '{output_file}' 文件中。")

except FileNotFoundError:
    print(f"错误：找不到文件 '{input_file}'。请确保该文件存在于当前工作目录中。")
except Exception as e:
    print(f"发生了一个错误：{e}")