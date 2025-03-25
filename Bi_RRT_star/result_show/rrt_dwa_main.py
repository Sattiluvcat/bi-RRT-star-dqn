import time

import pandas as pd
from matplotlib import pyplot as plt

from multi_planning.rrt_dwa import rrt_dwa
from single_planning.Bi_RRT import calc_path_length, plot_obs, plot_obs_rec


def run_rrt_dwa_times(run_times=100,environ=False):
    # 纯DWA算法时，设置跳出时间
    results = []

    for i in range(run_times):
        print(f"Run {i + 1}")
        start = [0, 5]
        goal = [120, -95]
        obstacle_list = [
            # 移动大障碍
            [25, -10, 10],
            [90, -60, 10],
            [45, -70, 12],
            [80, -15, 14],
            [40, -45, 80, -30],
            # L型障碍
            # [78,-55,79,-25],
            # [45,-55,78,-56],
            # # [25, -20, 12],
            # [110, -50, 14],
            # [30, -75, 16],
            # [65, -5, 10]
        ]
        start_time_cal = time.time()
        try:
            path, rrt_path,stagnation_count,end_time = rrt_dwa(start, goal, obstacle_list,environ)
        # end_time = time.time()
            print("over-over")
            if path is not None:
                run_time = end_time - start_time_cal
                path_length = calc_path_length(path)
                results.append([run_time, path_length, stagnation_count])
            else:
                results.append([None, None, None])
        except Exception as e:
            print("Error:", e)
            results.append([None, None, None])

    df = pd.DataFrame(results, columns=['Run Time (s)', 'Path Length (m)', 'Stagnation Count'])
    df.to_excel('rrt_dwa_L_results.xlsx', index=False)
    print("Results saved to rrt_dwa_L_results.xlsx")




if __name__ == "__main__":


    # set为True表示大障碍物环境，False表示L型障碍
    run_rrt_dwa_times(100,True)

    # plt.plot(start[0], start[1], "xr", label="start")
    # plt.plot(goal[0], goal[1], "xg", label="goal")
    # plt.axis("equal")
    # plt.axis([0, 120, -105, 15])
    # ss=0
    # s0s=0
    # for obs in obstacle_list:
    #     if len(obs) == 3:
    #         plot_obs(obs[0], obs[1], obs[2],"b",ss)
    #         ss+=1
    #     elif len(obs) == 4:
    #         plot_obs_rec(obs[0], obs[1], obs[2], obs[3],"y",s0s)
    #         s0s+=1
    # plt.xlabel("x/m")
    # plt.ylabel("y/m")
    # plt.legend(loc="upper right")
    # plt.show()

    # path, dwa, x, goal, obstacles, config, rrt_path = rrt_dwa(start, goal, obstacle_list, True)

    # if path is not None:
    #     print("Path found and animation saved.")
    # else:
    #     print("No path found")
