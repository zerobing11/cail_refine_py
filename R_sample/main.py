import numpy as np
# 从我们创建的工具文件中导入所需的函数
from sphere_sampling_utils import sample_circle_on_unit_sphere, plot_all_results_in_one_view,write_ply_file

# --- 主执行逻辑 ---
if __name__ == '__main__':
    rvec_normalized = np.array([0.5691949129696117, -0.5874995151909439, 0.5752055899414765])
    N_samples = 100  # 每个圆采样的点数
    # theta采样的范围与步长
    center_r_angle = 120.5373004596848
    r_angles_to_sample_deg = [center_r_angle + i for i in np.arange(-1, 1, 0.2)]
    r_angles_to_sample = [np.deg2rad(angle) for angle in r_angles_to_sample_deg]
    #旋转轴点存储txt与存储变量
    temp_rotations_file = "rotations_temp.txt"
    # all_sampled_points = []
    rotation_count = 0 #计数
    # =========================================================================
    # 第一步：采样所有旋转，并写入临时文件
    # =========================================================================
    print("--- 第1步: 开始采样旋转并写入临时文件 ---")
    with open(temp_rotations_file, 'w') as f:
        # 第一层循环，对轴角的轴偏角进行采样
        for x_angle in np.arange(0, 2, 0.2):
            print(f"处理角度: {x_angle}°")
            # 第二层循环，对轴角的角度进行采样
            for center_r_angle in r_angles_to_sample:
                print(f"  正在采样 scale_factor: {center_r_angle:.4f}...")
                b_points = sample_circle_on_unit_sphere(rvec_normalized, x_angle, N_samples, center_r_angle)
                # all_sampled_points.extend(b_points)
                # 将采样到的旋转向量写入文件
                for point in b_points:
                    # 保留八位小数
                    line = f"{point[0]:.8f},{point[1]:.8f},{point[2]:.8f}"
                    f.write(line + '\n')
                    rotation_count += 1
    print(f"旋转采样完成！共 {rotation_count} 个旋转向量已写入 '{temp_rotations_file}'。")
    # 写入PLY文件
    # write_ply_file('sampled_points.ply', all_sampled_points)

    # =========================================================================
    # 第二步：采样平移，并与文件中的旋转组合，生成最终输出
    # =========================================================================
    # 平移采样参数
    t_initial = np.array([0.105757 ,0.0302148 ,0.0426588])
    t__range_x = 0.08
    t_step_x = 0.01
    t__range_y = 0.015
    t_step_y = 0.005
    t__range_z = 0.03
    t_step_z = 0.005

    # 生成平移向量列表 (这个列表很小，可以安全地放在内存中)
    sampled_translations = []

    for x in np.arange(t_initial[0] - t__range_x, t_initial[0] + t_step_x, t_step_x):
        for y in np.arange(t_initial[1] - t__range_y, t_initial[1] + t__range_y + t_step_y, t_step_y):
            for z in np.arange(t_initial[2] - t__range_z, t_initial[2] + t_step_z, t_step_z):
                sampled_translations.append(np.array([x, y, z]))
    print(f"平移采样完成，共生成 {len(sampled_translations)} 个平移向量。")

    # 读取临时旋转文件，进行组合并写入最终文件
    final_output_file = "sampled_poses_6dof.txt"
    pose_count = 0

    with open(temp_rotations_file, 'r') as f_rot, open(final_output_file, 'w') as f_out:
        f_out.write("# Format: rvec_x, rvec_y, rvec_z, t_x, t_y, t_z\n")

        # 逐行读取旋转文件
        for rvec_line in f_rot:
            # 读取旋转向量的字符串
            rvec_line_clean = rvec_line.strip()
            for tvec in sampled_translations:
                #保留8位小数
                tvec_line = f"{tvec[0]:.8f},{tvec[1]:.8f},{tvec[2]:.8f}"
                # 拼接旋转和平移的字符串
                final_line = rvec_line_clean + ',' + tvec_line
                f_out.write(final_line + '\n')
                pose_count += 1

            if pose_count % 100000 == 0 and pose_count > 0:
                print(f"已组合并写入 {pose_count} 个位姿...")

    print(f"\n处理完成！")
    print(f"总共 {pose_count} 个6DoF位姿已全部写入 '{final_output_file}'。")
