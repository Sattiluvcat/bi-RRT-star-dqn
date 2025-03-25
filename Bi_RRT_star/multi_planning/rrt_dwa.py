import os
import shutil
import time

import numpy as np
from matplotlib import pyplot as plt, animation

from single_planning.Bi_RRT import check_collision, plot_obs, plot_obs_rec, check_point_in_obs
from single_planning.Bi_RRT_star import Bi_RRT_star_plan
from single_planning.DWA import DWA
from utils.node import Node

os.makedirs('plots', exist_ok=True)


def update_obstacles(obstacles, set_environ=False):
    for obs in obstacles:
        # set为大障碍物环境
        if set_environ and len(obs) == 4:
            obs[0] -= 1
            obs[2] -= 1
        # 不 set 为 L 型障碍环境
        elif not set_environ and len(obs) == 3:
            obs[0] += np.random.uniform(0, 1.4)
            obs[1] += np.random.uniform(0, 1.4)


def rrt_dwa(start, goal_last, obstacles, set_environ=False):
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
        'dt': 1,  # Reduce time step
        'predict_time': 7.0,  # Reduce prediction time

        'obs_cost_gain': 1.0,  # 障碍物cost的权重
        'to_goal_cost_gain': 0.03,
        'speed_cost_gain': 1.0,
        # 机器人半径 没有用到的参数 可以改为到障碍物的距离
        'robot_radius': 1.0,
        # 向量场，此处恒为0
        'field_gain': 1.0
    }

    x = np.array([start[0], start[1], 0.0, 0.0, 0.0])
    path = [x[:2].copy()]
    rrt_path, end_time = Bi_RRT_star_plan(start, goal_last, obstacles)
    print("rrtpath:", rrt_path)

    # plt.plot([x for (x, y) in rrt_path], [y for (x, y) in rrt_path], 'r')
    # for obs in obstacles:
    #     if len(obs) == 3:
    #         plot_obs(obs[0], obs[1], obs[2])
    #     elif len(obs) == 4:
    #         plot_obs_rec(obs[0], obs[1], obs[2], obs[3])
    # plt.show()

    if rrt_path is None:
        return None

    dwa = DWA(config, None)
    rrt_index = 0

    goal = np.array(rrt_path[rrt_index])

    # plt.plot(x[0], x[1], "ob")
    # plt.axis([-10, 130, -105, 15])
    i = 0
    sum_time = 0.0
    stagnation_count = 0
    while rrt_index <= len(rrt_path):
        # TODO 增加目标权重-->这里选择不增加目标权重，因为重新生成路径之后
        #  最开始应该是用rrt走的，以后也是一样的循环，没办法用到dwa方法；
        #   除非在快要进入复杂环境时增加，但感觉也没什么用

        # 原逻辑
        if rrt_index == len(rrt_path):
            goal = np.array(rrt_path[-1])
        else:
            goal = np.array(rrt_path[rrt_index])
        print("goal:", goal)
        print("rrt_index:", rrt_index)
        # 与下方找到最近路径点函数保持对应关系（不能比函数的参数大）
        while np.linalg.norm(x[:2] - goal) >= 8:
            # 特别的限定函数，避免绕圈
            # if path[-1][0] > goal[0]:
            #     rrt_index += 1
            #     break
            print("in here")
            update_obstacles(obstacles, set_environ)  # Update the positions of the dynamic obstacles
            for obs in obstacles:
                if len(obs) == 3:
                    plot_obs(obs[0], obs[1], obs[2])
                elif len(obs) == 4:
                    plot_obs_rec(obs[0], obs[1], obs[2], obs[3], "y")

            # Check for collision along the line to the goal
            if not check_collision(Node(x[0], x[1]),
                                   Node(x[0] + 18 * np.cos(np.arctan2(goal[1] - x[1], goal[0] - x[0])),
                                        x[1] + 18 * np.sin(np.arctan2(goal[1] - x[1], goal[0] - x[0]))), obstacles):
                # No collision, move directly towards the goal
                direction = np.arctan2(goal[1] - x[1], goal[0] - x[0])
                speed = config['max_speed']
                steering_angle = direction - x[2]
                x = dwa.motion(x, [speed, steering_angle])
                print("out!")
                print("rrt_index:", rrt_index)
            else:
                # Collision detected, use DWA
                print("I'm in!")
                start_time = time.time()
                u, trajectory, obs_lv = dwa.plan(x, goal, obstacles)
                x = dwa.motion(x, u)

                end_time = time.time()
                if isinstance(trajectory[0], float) or obs_lv > 0.4:
                    # stop_status=True
                    sum_time += end_time - start_time
                    if sum_time >= 4:
                        print("停滞时间过长，生成RRT路径中···")
                        # 已经停滞，相当于起点，不用考虑转弯角度什么的
                        tmp_rrt_path, plan_time = Bi_RRT_star_plan(x[:2], goal_last, obstacles)
                        # print("tmp_rrt_path:", tmp_rrt_path)
                        if tmp_rrt_path is not None:
                            rrt_path = tmp_rrt_path
                            rrt_index = 0
                        else:
                            return None, None, None, None
                        sum_time = 0.0
                        stagnation_count += 1
                    break
            print("printA")
            path.append(x[:2].copy())

            # Plot the path
            plt.plot([x for (x, y) in rrt_path], [y for (x, y) in rrt_path], 'r--')
            for p in path:
                plt.plot(p[0], p[1], ".g")
            plt.axis([0, 120, -105, 15])

            plt.plot(np.array(rrt_path[-1])[0], np.array(rrt_path[-1])[1], "xk")
            plt.plot(goal[0], goal[1], "xk")
            # 保存plot到集合中
            plt.title(f'Plot {i}')
            plt.savefig(f'plots/plot_{i:03d}.png')
            plt.show()

            i += 1

            if not obstacles_in_front(x, goal, obstacles):
                break
            if np.linalg.norm(x[:2] - np.array(rrt_path[-1])) < 10:  # Check if close to the final goal
                print("pintB")
                break
            if check_point_in_obs(goal, obstacles):  # Check if the goal is inside an obstacle
                break
        # 最后5米没有障碍物直接到达 逻辑上有点问题 但是不重要
        if np.linalg.norm(x[:2] - np.array(rrt_path[-1])) < 10 and not obstacles_in_front(x, goal, obstacles):
            print("pintC")
            # while np.linalg.norm(x[:2] - goal_last) >= 1:
            #     direction = np.arctan2(goal[1] - x[1], goal[0] - x[0])
            #     speed = config['max_speed']
            #     steering_angle = direction - x[2]
            #     x = dwa.motion(x, [speed, steering_angle])
            #     path.append(x[:2].copy())
            # plt.plot([x for (x, y) in rrt_path], [y for (x, y) in rrt_path], 'r--')
            # for p in path:
            #     plt.plot(p[0], p[1], ".g")
            # plt.axis([0, 120, -105, 15])
            break
        last_goal_index = rrt_index
        if rrt_index == len(rrt_path):
            rrt_index = len(rrt_path)
        else:
            rrt_index = find_nearest_point_index(x, rrt_path, last_goal_index, obstacles)
    # 最后5米没有障碍物直接到达
    if not obstacles_in_front(path[-1], rrt_path[-1], obstacles) and np.linalg.norm(path[-1] - rrt_path[-1]) < 10:
        print("pintD")
        # break
        # while np.linalg.norm(x[:2] - goal_last) >= 1:
        #     direction = np.arctan2(goal[1] - x[1], goal[0] - x[0])
        #     speed = config['max_speed']
        #     steering_angle = direction - x[2]
        #     x = dwa.motion(x, [speed, steering_angle])
        #     path.append(x[:2].copy())
        #     # plt.plot([x for (x, y) in rrt_path], [y for (x, y) in rrt_path], 'r--')
        #     # for p in path:
        #     #     plt.plot(p[0], p[1], ".g")
        #     # plt.axis([0, 120, -105, 15])
        path.append(rrt_path[-1])
    print("pintE")
    # for p in path:
    #     plt.plot(p[0], p[1], ".g")
    # plt.axis([0, 120, -105, 15])
    # plt.plot(np.array(rrt_path[-1])[0], np.array(rrt_path[-1])[1], "xk")
    # plt.plot([x for (x, y) in rrt_path], [y for (x, y) in rrt_path], 'y')
    # plt.plot(goal[0], goal[1], "xr")
    # # 保存plot到集合中
    # plt.title(f'Plot {i}')
    # plt.savefig(f'plots/plot_{i:03d}.png')

    # fig = plt.figure()
    # images = []
    # for j in range(i):
    #     img = plt.imread(f'plots/plot_{j:03d}.png')
    #     im = plt.imshow(img, animated=True)
    #     images.append([im])
    #
    # # 使用 ArtistAnimation 创建动画
    # ani = animation.ArtistAnimation(fig, images, interval=200, blit=True)
    # # 保存动画为 GIF 文件
    # ani.save('output_animation.gif', writer='pillow')
    # shutil.rmtree('plots')

    print("path:", path)

    return path, rrt_path, stagnation_count, end_time


def obstacles_in_front(x, goal, obstacles):
    x_pos = Node(x[0], x[1])
    # print("x_pos:", x_pos)  # Extract the position part of x
    if check_collision(x_pos, goal, obstacles):
        return True
    return False


def find_nearest_point_index(x, path, start_index, obstacles):
    min_dist = float('inf')
    # 相距大于8m的点才考虑 若与当前目标点相距小于8m则自动从下一目标点开始检索
    max_dist = 8
    nearest_index = start_index
    for i in range(start_index, len(path)):
        dist = np.linalg.norm(x[:2] - path[i])
        print("dist:", dist,"i:",i,"min_dist:",min_dist)
        # TODO: 适用于第一种情况
        # if min_dist > dist >= max_dist and not check_point_in_obs(path[i], obstacles):
        if min_dist > dist >= max_dist:
            min_dist = dist
            nearest_index = i
    print("nearest_index:", nearest_index)
    return nearest_index
