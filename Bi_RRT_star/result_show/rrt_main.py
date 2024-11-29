# rrt样例
import time

from matplotlib import pyplot as plt

from single_planning.Bi_RRT import path_score, plot_obs_rec, plot_obs, RRT_plan, re_obs
from single_planning.Bi_RRT_star import Bi_RRT_star_plan

if __name__ == "__main__":
    start_time = time.time()
# 静态障碍物
    start = [8.77650817669928, 4.951398874633014]
    goal = [249.09851929917932, -195.2040752498433]
    obstacle_list = [[33.77685192972422, -52.90446031652391, 7.5], [46.19124796241522, -77.85628066305071, 9.0],
                     [80.34613360464573, -43.54909089393914, 7.5], [111.38825733587146, -74.74188929889351, 7.5],
                     [80.16235236078501, 9.456780110485852, 4.5], [139.31754435040057, -12.368450773879886, 4.5],
                     [207.6151233687997, 3.4003808852285147, 15.0], [111.59657261520624, -127.13194767106324, 15.0],
                     [307.5552379246801, -6.496737029403448, 15.0], [182.18576977215707, -97.71181975770742, 15.0],
                     [232.431648472324, -58.93625687714666, 220, -70], [76.79213218018413, -161.22955951932818, 7.5],
                     [168.2150300759822, -149.5354239968583, 7.5], [265.8879858329892, -127.0088061131537, 7.5],
                     [325.93784911744297, -71.36904122401029, 7.5], [284.0254806391895, -45.27248460613191, 4.5]]

    obstacle_list = re_obs(obstacle_list)
# 路径规划
    # Bi-RRT方法
    # path = RRT_plan(start, goal, obstacle_list)

    # Bi-RRT*方法
    path = Bi_RRT_star_plan(start, goal, obstacle_list)

    print(path)
# 画图
    obstacle = obstacle_list
    plt.plot(start[0], start[1], "xk")
    plt.plot(goal[0], goal[1], "xk")
    plt.axis("equal")
    plt.axis([0, 260, -200, 10])

    for obs in obstacle:
        if len(obs) == 3:
            plot_obs(obs[0], obs[1], obs[2])
        elif len(obs) == 4:
            plot_obs_rec(obs[0], obs[1], obs[2], obs[3])

    plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r')

    plt.show()
    end_time = time.time()
    all_time = end_time - start_time
    path_score(path, all_time, obstacle_list)
