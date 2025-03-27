# vfrrt*结合dwa的规划主函数，新增产生向量场函数，rrt路径由VF_Bi_RRT_star_plan函数生成
import os
import shutil
import time

import numpy as np
from matplotlib import pyplot as plt, animation

from multi_planning.rrt_dwa import update_obstacles, obstacles_in_front, find_nearest_point_index
from single_planning.Bi_RRT import check_collision, plot_obs, plot_obs_rec
from single_planning.VF_Bi_RRT_star import VF_Bi_RRT_star_plan
from single_planning.DWA import DWA
from utils.node import Node

os.makedirs('plots', exist_ok=True)

# 向量大小

def generate_vector_field(x_range, y_range, spacing):
    x = np.arange(x_range[0], x_range[1], spacing)
    y = np.arange(y_range[0], y_range[1], spacing)
    X, Y = np.meshgrid(x, y)

    # 计算中心点——设置一个圆形向量场
    center_x = (x_range[0] + x_range[1]) / 2
    center_y = (y_range[0] + y_range[1]) / 2

    # 计算向量场
    U = -(Y - center_y)
    V = X - center_x

    # 归一化
    magnitude = np.sqrt(U ** 2 + V ** 2)
    U = U / magnitude
    V = V / magnitude

    # 引入变化规律
    U = U * (1 + 0.5 * np.sin(X / 10) * np.cos(Y / 10))
    V = V * (1 + 0.5 * np.sin(X / 10) * np.cos(Y / 10))

    return X, Y, U, V


def vf_rrt_dwa(start, goal, obstacles):
    config = {
        'max_speed': 2.0,  # Reduce maximum speed
        'min_speed': -2.0,  # Reduce minimum speed
        'max_yawrate': 40.0 * np.pi / 180.0,  # Reduce maximum yaw rate
        'max_accel': 1,  # Reduce maximum acceleration
        'max_dyawrate': 10.0 * np.pi / 180.0,  # Reduce maximum change in yaw rate
        'v_reso': 0.15,  # 速度的分辨率（同下）
        'yawrate_reso': 0.15 * np.pi / 180.0,  # 旋转速率的分辨率（步长——每次增加这么多）
        'dt': 1.0,  # Reduce time step
        'predict_time': 5.0,  # Reduce prediction time
        'obs_cost_gain': 0.05,  # 障碍物cost的权重
        'to_goal_cost_gain': 0.5,
        'speed_cost_gain': 1.0,
        'robot_radius': 1.0,    # 机器人半径
        # 相对非vf的dwa增加的参数，考虑向量方向影响
        'field_gain': 0.5
    }

    # 生成向量场——范围稍大于图形范围
    X, Y, U, V = generate_vector_field((-20, 280), (-220, 20), 1)
    vector_feild = (X, Y, U, V)

    dwa = DWA(config,vector_feild)
    x = np.array([start[0], start[1], 0.0, 0.0, 0.0])
    path = [x[:2].copy()]

    vf_rrt_path = VF_Bi_RRT_star_plan(start, goal, obstacles, vector_feild)
    print("vfrrtpath:", vf_rrt_path)

    plt.plot([x for (x, y) in vf_rrt_path], [y for (x, y) in vf_rrt_path], 'r')
    for obs in obstacles:
        if len(obs) == 3:
            plot_obs(obs[0], obs[1], obs[2])
        elif len(obs) == 4:
            plot_obs_rec(obs[0], obs[1], obs[2], obs[3])
    plt.show()

    if vf_rrt_path is None:
        return None

    rrt_index = 0

    goal = np.array(vf_rrt_path[rrt_index])

    plt.plot(x[0], x[1], "ob")
    plt.axis([0, 260, -200, 10])
    i = 0
    sum_time = 0.0
    while rrt_index <= len(vf_rrt_path):
        if rrt_index == len(vf_rrt_path):
            goal = np.array(vf_rrt_path[-1])
        else:
            goal = np.array(vf_rrt_path[rrt_index])
        # print("goal:", goal[0], goal[1])
        # 与下方找到最近路径点函数保持对应关系（不能比函数的参数大）
        while np.linalg.norm(x[:2] - goal) >= 3:
            update_obstacles(obstacles)  # Update the positions of the dynamic obstacles
            for obs in obstacles:
                if len(obs) == 3:
                    plot_obs(obs[0], obs[1], obs[2])
                elif len(obs) == 4:
                    plot_obs_rec(obs[0], obs[1], obs[2], obs[3])
            # print("now")

            # Check for collision along the line to the goal，前方 10m 有障碍物则切换
            if not check_collision(Node(x[0], x[1]),
                                   Node(x[0] + 10 * np.cos(np.arctan2(goal[1] - x[1], goal[0] - x[0])),
                                        x[1] + 10 * np.sin(np.arctan2(goal[1] - x[1], goal[0] - x[0]))), obstacles):
                # No collision, move directly towards the goal
                # print("I'm out!")
                direction = np.arctan2(goal[1] - x[1], goal[0] - x[0])
                speed=config['max_speed']
                steering_angle = direction - x[2]
                x = dwa.motion(x, [speed, steering_angle])
            else:
                # Collision detected, use DWA
                print("I'm in!")
                start_time = time.time()
                u, _ = dwa.plan(x, goal, obstacles)
                x = dwa.motion(x, u)

                end_time = time.time()
                if u == [0.0, 0.0]:
                    # stop_status = True
                    sum_time += end_time - start_time
                    if sum_time >= 2:
                        print("停滞时间过长，生成RRT路径中···")
                        # 已经停滞，相当于起点，不用考虑转弯角度什么的
                        tmp_rrt_path = VF_Bi_RRT_star_plan(x[:2], goal, obstacles,vector_feild)
                        if tmp_rrt_path is not None:
                            vf_rrt_path = tmp_rrt_path + vf_rrt_path[rrt_index + 1:]
                            rrt_index = 0
                    sum_time = 0.0
                    break

            path.append(x[:2].copy())

            # Plot the path
            plt.plot([x for (x, y) in vf_rrt_path], [y for (x, y) in vf_rrt_path], 'y')
            # print("path:", path)
            for p in path:
                plt.plot(p[0], p[1], ".g")
            plt.axis([0, 260, -200, 10])
            # 画向量场
            plt.quiver(X[::10, ::10], Y[::10, ::10], U[::10, ::10], V[::10, ::10], color='b')

            plt.plot(np.array(vf_rrt_path[-1])[0], np.array(vf_rrt_path[-1])[1], "xk")
            plt.plot(goal[0], goal[1], "xr")
            # 保存plot到集合中
            plt.title(f'Plot {i}')
            plt.savefig(f'plots/plot_{i:03d}.png')

            plt.show()
            # plt.close()
            i += 1

            if not obstacles_in_front(x, goal, obstacles):
                break
            if np.linalg.norm(x[:2] - np.array(vf_rrt_path[-1])) < 10:  # Check if close to the final goal
                break
        # 最后 4 米没有障碍物直接到达 逻辑上有点问题 但是不重要
        if np.linalg.norm(x[:2] - np.array(vf_rrt_path[-1])) < 4 and not obstacles_in_front(x, goal, obstacles):
            break
        last_goal_index = rrt_index
        if rrt_index == len(vf_rrt_path):
            rrt_index = len(vf_rrt_path)
        else:
            rrt_index = find_nearest_point_index(x, vf_rrt_path, last_goal_index,obstacles)
        # print("rrtindex:", rrt_index)

    # 最后十米没有障碍物直接到达
    if not obstacles_in_front(path[-1], vf_rrt_path[-1], obstacles) and np.linalg.norm(path[-1] - vf_rrt_path[-1]) < 5:
        path.append(vf_rrt_path[-1])

    for p in path:
        plt.plot(p[0], p[1], ".g")
    # Plot the RRT path
    # ax.plot([x for (x, y) in rrt_path], [y for (x, y) in rrt_path], 'r')
    plt.axis([0, 260, -200, 10])
    plt.plot(np.array(vf_rrt_path[-1])[0], np.array(vf_rrt_path[-1])[1], "xk")
    plt.plot([x for (x, y) in vf_rrt_path], [y for (x, y) in vf_rrt_path], 'y')
    plt.plot(goal[0], goal[1], "xr")
    # 保存plot到集合中
    plt.title(f'Plot {i}')
    plt.savefig(f'plots/plot_{i:03d}.png')

    fig = plt.figure()
    images = []
    for j in range(i):
        img = plt.imread(f'plots/plot_{j:03d}.png')
        im = plt.imshow(img, animated=True)
        images.append([im])

    # 使用 ArtistAnimation 创建动画
    ani = animation.ArtistAnimation(fig, images, interval=200, blit=True)
    # 保存动画为 GIF 文件
    ani.save('output_animation.gif', writer='pillow')
    shutil.rmtree('plots')

    # print("path:", path)

    return path, dwa, x, goal, obstacles, config, vf_rrt_path
