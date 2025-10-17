# sphere_sampling_utils.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def sample_circle_on_unit_sphere(rvec_normalized, angle_deg, num_samples, center_r_angle=1.0):
    """
    在单位球(半径为1)上，围绕一个给定的归一化向量A，在固定夹角为angle_deg的圆上均匀采样，并且
    乘以一个固定值scale_factor来缩放每个采样点。

    参数:
    rvec_normalized (np.ndarray): 球面上A点的归一化三维坐标向量。
    angle_deg (float): OA向量与OB向量的夹角（单位：度）。
    num_samples (int): 需要采样的点的数量。
    center_r_angle (float): 用来缩放每个采样点的标量因子，默认值为1.0。

    返回:
    list of np.ndarray: 圆上采样点的坐标列表，所有点都乘以center_r_angle。
    """
    vec_A = rvec_normalized
    theta_rad = np.deg2rad(angle_deg)
    w = vec_A

    world_z = np.array([0, 0, 1])
    if np.all(np.isclose(w, world_z)) or np.all(np.isclose(w, -world_z)):
        world_ref = np.array([1, 0, 0])
    else:
        world_ref = world_z

    u = np.cross(world_ref, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)

    sampled_points = []
    circle_radius = np.sin(theta_rad)
    dist_to_center = np.cos(theta_rad)
    circle_center = dist_to_center * w

    for i in range(num_samples):
        phi = 2 * np.pi * i / num_samples
        vec_B = circle_center + circle_radius * (np.cos(phi) * u + np.sin(phi) * v)

        # 在每个采样点乘以center_r_angle
        vec_B_scaled = center_r_angle * vec_B

        sampled_points.append(vec_B_scaled)

    return sampled_points

def write_ply_file(filename, points):
    """
    将三维点写入PLY文件。
    参数:
    filename (str): 输出文件名。
    points (list of np.ndarray): 包含要写入的点的列表。
    """
    if not points:
        print("警告: 没有点可写入PLY文件。")
        return

    # 将点列表转换为Numpy数组
    points_array = np.vstack(points)
    num_points = points_array.shape[0]

    # PLY文件头
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
end_header
"""
    # 写入文件
    with open(filename, 'w') as f:
        f.write(header)
        for point in points_array:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
    print(f"已成功将 {num_points} 个点保存到 '{filename}'。")


def plot_all_results_in_one_view(point_a, all_results_map):
    """
    将所有采样结果绘制在同一个三维视图中，并按角度为点着色。

    参数:
    point_a (np.ndarray): 原始的、归一化的A点向量。
    all_results_map (dict): 包含所有结果的嵌套字典，
                            格式为: {angle: {scale: [points]}, ...}
    """
    print("正在生成统一视图...")
    fig = plt.figure(figsize=(13, 11))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 确定参考球体的大小（找到所有scale_factor中的最大值）
    max_radius = 0
    for inner_map in all_results_map.values():
        max_radius = max(max_radius, max(inner_map.keys()))

    if max_radius > 0:
        u_sphere = np.linspace(0, 2 * np.pi, 100)
        v_sphere = np.linspace(0, np.pi, 50)
        x_sphere = max_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
        y_sphere = max_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
        z_sphere = max_radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.05, linewidth=0)

    # 2. 准备颜色循环
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # 3. 遍历数据并按角度上色
    for i, (angle, inner_map) in enumerate(all_results_map.items()):
        color = colors[i % len(colors)]
        is_first_plot_for_this_angle = True

        for scale, points in inner_map.items():
            points_B = np.array(points)
            label = f'Angle = {angle}°' if is_first_plot_for_this_angle else None

            ax.scatter(points_B[:, 0], points_B[:, 1], points_B[:, 2],
                       color=color, s=15, label=label)

            point_A_scaled = point_a * scale
            ax.scatter(point_A_scaled[0], point_A_scaled[1], point_A_scaled[2],
                       color=color, marker='*', s=180, edgecolors='black')

            is_first_plot_for_this_angle = False

    # 4. 设置图像属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('All Sampled Circles on a Single Plot')
    ax.legend()
    ax.set_box_aspect([1, 1, 1])

    plt.show()