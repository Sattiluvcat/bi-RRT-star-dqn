import os
import shutil

import numpy as np
from matplotlib import pyplot as plt, animation

from single_planning.Bi_RRT import check_collision, plot_obs, plot_obs_rec
from single_planning.Bi_RRT_star import Bi_RRT_star_plan
from single_planning.DWA import DWA
from utils.node import Node

os.makedirs('plots', exist_ok=True)


def update_obstacles(obstacles):
    for obs in obstacles:
        obs[0] += np.random.uniform(0, 0.5)  # Randomly move the obstacle in x direction
        obs[1] += np.random.uniform(0, 0.5)  # Randomly move the obstacle in y direction


def rrt_dwa(start, goal_last, obstacles):
    config = {
        'max_speed': 2.0,  # Reduce maximum speed
        'min_speed': -2.0,  # Reduce minimum speed
        'max_yawrate': 40.0 * np.pi / 180.0,  # Reduce maximum yaw rate
        'max_accel': 1,  # Reduce maximum acceleration
        'max_dyawrate': 10.0 * np.pi / 180.0,  # Reduce maximum change in yaw rate
        'v_reso': 0.15,  # 速度的分辨率（同下）
        'yawrate_reso': 0.15 * np.pi / 180.0,  # 旋转速率的分辨率（步长——每次增加这么多）

        # dt与predict_time会影响DWA程序中的障碍物cost评价
        # （只取前1s是否会与障碍物发生碰撞进行判断）
        # 需要对应修改
        'dt': 1.0,  # Reduce time step
        'predict_time': 5.0,  # Reduce prediction time

        'to_goal_cost_gain': 2.0,
        'speed_cost_gain': 1.0,
        # 机器人半径 没有用到的参数 可以改为到障碍物的距离
        'robot_radius': 1.0
    }

    dwa = DWA(config)
    x = np.array([start[0], start[1], 0.0, 0.0, 0.0])
    path = [x[:2].copy()]
    rrt_path = Bi_RRT_star_plan(start, goal_last, obstacles)
    print("rrtpath:", rrt_path)

    plt.plot([x for (x, y) in rrt_path], [y for (x, y) in rrt_path], 'r')
    for obs in obstacles:
        if len(obs) == 3:
            plot_obs(obs[0], obs[1], obs[2])
        elif len(obs) == 4:
            plot_obs_rec(obs[0], obs[1], obs[2], obs[3])
    plt.show()

    if rrt_path is None:
        return None

    rrt_index = 0

    goal = np.array(rrt_path[rrt_index])

    plt.plot(x[0], x[1], "ob")
    plt.axis([0, 260, -200, 10])
    i = 0
    while rrt_index <= len(rrt_path):
        if rrt_index == len(rrt_path):
            goal = np.array(rrt_path[-1])
        else:
            goal = np.array(rrt_path[rrt_index])
        print("goal:",goal[0], goal[1])
        # 与下方找到最近路径点函数保持对应关系（不能比函数的参数大）
        while np.linalg.norm(x[:2] - goal) >= 8:
            update_obstacles(obstacles)  # Update the positions of the dynamic obstacles
            for obs in obstacles:
                if len(obs) == 3:
                    plot_obs(obs[0], obs[1], obs[2])
                elif len(obs) == 4:
                    plot_obs_rec(obs[0], obs[1], obs[2], obs[3])
            print("now")

            # Check for collision along the line to the goal
            if not check_collision(Node(x[0], x[1]),
                                   Node(x[0] + 25 * np.cos(np.arctan2(goal[1] - x[1], goal[0] - x[0])),
                                        x[1] + 25 * np.sin(np.arctan2(goal[1] - x[1], goal[0] - x[0]))), obstacles):
                # No collision, move directly towards the goal
                direction = np.arctan2(goal[1] - x[1], goal[0] - x[0])
                x[0] += config['max_speed'] * np.cos(direction)
                x[1] += config['max_speed'] * np.sin(direction)
            else:
                # Collision detected, use DWA
                print("I'm in!")
                u, _ = dwa.plan(x, goal, obstacles)
                x = dwa.motion(x, u)

            path.append(x[:2].copy())

            # Plot the path
            plt.plot([x for (x, y) in rrt_path], [y for (x, y) in rrt_path], 'y')
            print("path:", path)
            for p in path:
                plt.plot(p[0], p[1], ".g")
            plt.axis([0, 260, -200, 10])

            plt.plot(np.array(rrt_path[-1])[0], np.array(rrt_path[-1])[1], "xk")
            plt.plot(goal[0], goal[1], "xr")
            # 保存plot到集合中
            plt.title(f'Plot {i}')
            plt.savefig(f'plots/plot_{i:03d}.png')

            plt.show()
            # plt.close()
            i += 1

            # # 逻辑go on
            # if rrt_index == len(rrt_path):
            #     goal = np.array(rrt_path[-1]) + 10 * np.array([np.cos(x[2]), np.sin(x[2])])
            #     # goal = np.array(rrt_path[-1])
            # else:
            #     goal = np.array(rrt_path[rrt_index]) + 15 * np.array([np.cos(x[2]), np.sin(x[2])])
            #     # goal = np.array(rrt_path[rrt_index])

            if not obstacles_in_front(x, goal, obstacles):
                break
            if np.linalg.norm(x[:2] - np.array(rrt_path[-1])) < 10:  # Check if close to the final goal
                break
        # 最后十米没有障碍物直接到达 逻辑上有点问题 但是不重要
        if np.linalg.norm(x[:2] - np.array(rrt_path[-1])) < 10 and not obstacles_in_front(x, goal, obstacles):
            break
        last_goal_index = rrt_index
        if rrt_index == len(rrt_path):
            rrt_index = len(rrt_path)
        else:
            rrt_index = find_nearest_point_index(x, rrt_path, last_goal_index)
        print("rrtindex:", rrt_index)

    # 最后十米没有障碍物直接到达
    if not obstacles_in_front(path[-1], rrt_path[-1], obstacles) and np.linalg.norm(path[-1] - rrt_path[-1]) < 10:
        path.append(rrt_path[-1])

    for p in path:
        plt.plot(p[0], p[1], ".g")
    # Plot the RRT path
    # ax.plot([x for (x, y) in rrt_path], [y for (x, y) in rrt_path], 'r')
    plt.axis([0, 260, -200, 10])
    plt.plot(np.array(rrt_path[-1])[0], np.array(rrt_path[-1])[1], "xk")
    plt.plot([x for (x, y) in rrt_path], [y for (x, y) in rrt_path], 'y')
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

    print("path:", path)

    return path, dwa, x, goal, obstacles, config, rrt_path


def obstacles_in_front(x, goal, obstacles):
    x_pos = Node(x[0], x[1])
    print("x_pos:", x_pos)  # Extract the position part of x
    if check_collision(x_pos, goal, obstacles):
        return True
    return False


def find_nearest_point_index(x, path, start_index):
    min_dist = float('inf')
    # 相距大于10m的点才考虑 若与当前目标点相距小于10m则自动从下一目标点开始检索
    max_dist=10
    nearest_index = start_index
    for i in range(start_index, len(path)):
        dist = np.linalg.norm(x[:2] - path[i])
        if min_dist > dist >= max_dist:
            min_dist = dist
            nearest_index = i
    return nearest_index
