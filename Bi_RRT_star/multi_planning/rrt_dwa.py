import os
import shutil

import numpy as np
from matplotlib import pyplot as plt, animation

from single_planning.Bi_RRT import check_collision, plot_obs, plot_obs_rec
from single_planning.DWA import DWA
from single_planning.Bi_RRT_star import Bi_RRT_star_plan
from utils.node import Node
from utils.plot2modes import plot_obstacle

os.makedirs('plots', exist_ok=True)

def update_obstacles(obstacles):
    for obs in obstacles:
        obs[0] += np.random.uniform(-0.5, 0.5)  # Randomly move the obstacle in x direction
        obs[1] += np.random.uniform(-0.5, 0.5)  # Randomly move the obstacle in y direction


def rrt_dwa(start, goal, obstacles):
    config = {
        'max_speed': 5.0,  # Reduce maximum speed
        'min_speed': -2.5,  # Reduce minimum speed
        'max_yawrate': 20.0 * np.pi / 180.0,  # Reduce maximum yaw rate
        'max_accel': 1,  # Reduce maximum acceleration
        'max_dyawrate': 20.0 * np.pi / 180.0,  # Reduce maximum change in yaw rate
        'v_reso': 0.2,  # Increase resolution for speed
        'yawrate_reso': 0.2 * np.pi / 180.0,  # Increase resolution for yaw rate
        'dt': 1.0,  # Reduce time step
        'predict_time': 5.0,  # Reduce prediction time
        'to_goal_cost_gain': 1.0,
        'speed_cost_gain': 1.0,
        # 取1比取10效果更好 这是反着来的
        'robot_radius': 1.0
    }

    dwa = DWA(config)
    x = np.array([start[0], start[1], 0.0, 0.0, 0.0])
    path = [x[:2].copy()]
    rrt_path = Bi_RRT_star_plan(start, goal, obstacles)
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


    # fig, ax = plt.subplots()

    goal = np.array(rrt_path[-1])


    plt.plot(x[0], x[1], "ob")
    plt.axis([0, 260, -200, 10])
    i=0
    while rrt_index <= len(rrt_path):
        if rrt_index==len(rrt_path):
            goal=np.array(rrt_path[-1])
        else:
            goal = np.array(rrt_path[rrt_index])
        while np.linalg.norm(x[:2] - goal) > 5:
            update_obstacles(obstacles)  # Update the positions of the dynamic obstacles
            for obs in obstacles:
                if len(obs) == 3:
                    plot_obs(obs[0], obs[1], obs[2])
                elif len(obs) == 4:
                    plot_obs_rec(obs[0], obs[1], obs[2], obs[3])

            print("Im in!")
            u, _ = dwa.plan(x, goal, obstacles)
            x = dwa.motion(x, u)
            path.append(x[:2].copy())

            # Plot the path
            print("path:",path)
            for p in path:
                plt.plot(p[0], p[1], ".g")
            # Plot the RRT path
            # ax.plot([x for (x, y) in rrt_path], [y for (x, y) in rrt_path], 'r')
            plt.axis([0, 260, -200, 10])
            plt.plot(np.array(rrt_path[-1])[0], np.array(rrt_path[-1])[1], "xk")
            plt.plot(goal[0], goal[1], "xr")
            # 保存plot到集合中
            plt.title(f'Plot {i}')
            plt.savefig(f'plots/plot_{i:03d}.png')

            plt.show()
            # plt.close()
            i+=1
            # 逻辑go on
            if rrt_index == len(rrt_path):
                goal = np.array(rrt_path[-1]) + 5 * np.array([np.cos(x[2]), np.sin(x[2])])
            else:
                goal = np.array(rrt_path[rrt_index]) + 15 * np.array([np.cos(x[2]), np.sin(x[2])])

            if not obstacles_in_front(x, goal, obstacles):
                break
            if np.linalg.norm(x[:2] - np.array(rrt_path[-1])) < 5:  # Check if close to the final goal
                break
        if np.linalg.norm(x[:2] - np.array(rrt_path[-1])) < 5 and not obstacles_in_front(x, goal, obstacles):
            break
        last_goal_index = rrt_index
        if rrt_index==len(rrt_path):
            rrt_index = len(rrt_path)
        else:
            rrt_index = find_nearest_point_index(x, rrt_path, last_goal_index+1)
        print("rrtindex:", rrt_index)

    # 最后五米没有障碍物直接到达 逻辑上有点问题 但是不重要
    if not obstacles_in_front(path[-1], rrt_path[-1], obstacles):
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
    nearest_index = start_index
    for i in range(start_index, len(path)):
        dist = np.linalg.norm(x[:2] - path[i])
        if dist < min_dist:
            min_dist = dist
            nearest_index = i
    return nearest_index
