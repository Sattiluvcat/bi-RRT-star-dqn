# rrt样例
import time
from cProfile import label

import pandas as pd
import pyttsx3
from matplotlib import pyplot as plt

from single_planning.Bi_RRT import calc_path_length, calc_triangle_deg, plot_obs, plot_obs_rec, RRT_plan
from single_planning.Bi_RRT_star import Bi_RRT_star_plan


def run_bi_rrt_multiple_times(start, goal, obstacle_list, num_runs=100):
    results = []
    all_paths = []
    mm=1000
    min_path=[]
    for i in range(num_runs):
        print("Run", i + 1)
        start_time = time.time()
        path,end_time = Bi_RRT_star_plan(start, goal, obstacle_list)
        # end_time = time.time()

        if path is not None:
            run_time = end_time - start_time
            path_length = calc_path_length(path)
            total_angle = 0
            max_angle = 0
            n = 0
            if len(path) >= 3:
                for j in range(len(path) - 2):
                    angle = abs(180 - calc_triangle_deg(path[j + 1], path[j], path[j + 2]))
                    if angle>75:
                        n+=1
                    total_angle += angle
                    max_angle = max(max_angle, angle)
            if path_length/250+max_angle/80<mm:
                min_path=path
                mm=path_length/250+max_angle/80
            results.append([run_time, path_length, total_angle, max_angle,n])
            all_paths.append(path)
        else:
            num_runs=num_runs+1

    df = pd.DataFrame(results, columns=['Run Time (s)', 'Path Length (m)', 'Total Angle (°)', 'Max Angle (°)','Sum'])
    df.to_excel('rrt_star_results_sum_pu.xlsx', index=False)
    print("Results saved to rrt_star_results_sum_pu.xlsx")

    # Plot all paths
    plt.figure()
    for i, path in enumerate(all_paths):
        plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r', alpha=0.3, label='all paths' if i == 0 else "")
    plt.plot([x for (x, y) in min_path], [y for (x, y) in min_path], 'g',label='best path')
    plt.plot(start[0], start[1], "xk")
    plt.plot(goal[0], goal[1], "xk")
    plt.axis("equal")
    plt.axis([0, 260, -210, 20])
    for obs in obstacle_list:
        if len(obs) == 3:
            plot_obs(obs[0], obs[1], obs[2])
        elif len(obs) == 4:
            plot_obs_rec(obs[0], obs[1], obs[2], obs[3])
    plt.legend()
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    plt.show()


if __name__ == "__main__":
    # ！多障碍环境！
    start = [8.77650817669928, 4.951398874633014]
    goal = [249.09851929917932, -195.2040752498433]
    obstacle_list = [
        [150, -50, 30],
        # [60, -110, 20],
        # [105, -40, 15],
        # [170, -25, 20],
        # [210, -60, 25],
        # [240, -130, 25],
        [70, -125, 25],
        # Rectangular obstacles
        [50, -45, 90, -5],
        # [100, -100, 140, -60],
        # [150, -150, 190, -110],
        [150, -160, 200, -110],
        # Additional obstacles along the direct line from start to goal
        # [60, -60, 10],
        # [120, -130, 20],
        # [180, -175, 15],
        # [240, -240, 20]
    ]

    # ！受困环境！
    # start = [100,-80]
    # goal = [210,-100]
    # obstacle_list = [
    #     # 四周的框
    #     # [8,44.5,248,45.5],
    #     # [8,-195.5,248,-194.5],
    #     # [7.5,-195,8.5,45],
    #     # [247.5,-195,248.5,45],
    #     # 横边
    #     [50,-49.5,150,-50.5],
    #     # [128,14.5,158,15.5],[188,14.5,248,15.5],
    #     [50,-149.5,150,-150.5],
    #     # [98,-15.5,158,-14.5],
    #     # [38,-45.5,68,-44.5],[98,-45.5,128,-44.5],[158,-45.5,188,-44.5],[218,-45.5,248,-44.5],
    #     # [38,-75.5,128,-74.5],[218,-75.5,248,-74.5],
    #     # [8,-105.5,38,-104.5],[68,-105.5,98,-104.5],[218,-105.5,248,-104.5],
    #     # [8,-135.5,68,-134.5],[98,-135.5,158,-134.5],[188,-135.5,218,-134.5],
    #
    #     # 竖边
    #     [149.5,-50,150.5,-150]
    #     # [67.5,15,68.5,45],[67.5,-75,68.5,-45],
    #     # [97.5,15,98.5,45],[97.5,-75,98.5,-15],[97.5,-135,98.5,-105],[97.5,-195,98.5,-165],
    #     # [127.5,-15,128.5,45],[127.5,-165,128.5,-105],
    #     # [157.5,-75,158.5,-45],[157.5,-165,158.5,-105],
    #     # [187.5,-195,188.5,-15],
    #     # [217.5,-15,218.5,15]
    # ]

    run_bi_rrt_multiple_times(start, goal, obstacle_list)

    # plt.plot([start[0], goal[0]], [start[1], goal[1]], "xk")
    # plt.plot(start[0], start[1], "xr")
    # plt.plot(goal[0], goal[1], "xg")
    # plt.axis("equal")
    # plt.axis([0, 260, -200, 20])
    # for obs in obstacle_list:
    #     if len(obs) == 3:
    #         plot_obs(obs[0], obs[1], obs[2])
    #     elif len(obs) == 4:
    #         plot_obs_rec(obs[0], obs[1], obs[2], obs[3])
    # plt.xlabel("x/m")
    # plt.ylabel("y/m")
    # plt.legend(["start","goal","obstacles"])
    # plt.show()

    engine=pyttsx3.init()
    engine.say("The path has been planned successfully")
    engine.runAndWait()
