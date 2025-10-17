import numpy as np

# 旋转轴 (单位向量)
axis = np.array([0.5691949129696117 ,-0.5874995151909439 ,0.5752055899414765 ])

# 旋转角度 (度)
angle_degrees = 120.5373004596848

# 将角度转换为弧度
angle_radians = np.deg2rad(angle_degrees)

# 旋转向量 (轴角形式)
rotation_vector = axis * angle_radians

# 输出结果，保留小数点后8位
rotation_vector_rounded = np.round(rotation_vector, 8)

print(f"旋转向量：{rotation_vector_rounded}")
