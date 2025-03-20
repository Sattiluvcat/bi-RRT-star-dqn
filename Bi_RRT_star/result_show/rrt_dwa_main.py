import time

import pandas as pd

from multi_planning.rrt_dwa import rrt_dwa
from single_planning.Bi_RRT import calc_path_length


def run_rrt_dwa_times(start_point, goal_point, obs_list, run_times=100,environ=False):
    # 纯DWA算法时，设置跳出时间
    results = []

    for i in range(run_times):
        print(f"Run {i + 1}")
        start_time_cal = time.time()
        path, dwa, x, goal_point, obstacles, config, rrt_path = rrt_dwa(start_point, goal_point, obs_list,environ)
        end_time = time.time()

        if path is not None:
            run_time = end_time - start_time_cal
            path_length = calc_path_length(path)
            stagnation_count = sum(1 for p in path if p == [0.0, 0.0])
            results.append([run_time, path_length, stagnation_count])
        else:
            results.append([None, None, None])

    df = pd.DataFrame(results, columns=['Run Time (s)', 'Path Length (m)', 'Stagnation Count'])
    df.to_excel('rrt_dwa_results.xlsx', index=False)
    print("Results saved to rrt_dwa_results.xlsx")




if __name__ == "__main__":
    start_time = time.time()
    start = [0,5]
    goal = [120,-95]
    obstacle_list = [
        # 移动大障碍
        [25,-10,10],
        [90,-60,12],
        [50,-70,18],
        [65,-20,15],
        [90,-45,150,-20],
        # L型障碍
        # [78,-55,79,-25],
        # [45,-55,78,-56],
        # # [25, -20, 12],
        # [110, -50, 14],
        # [30, -75, 16],
        # [65, -5, 10]
    ]

    # set为True表示大障碍物环境，False表示L型障碍
    run_rrt_dwa_times(start,goal,obstacle_list,100,True)

    # plt.plot(start[0], start[1], "xr", label="start")
    # plt.plot(goal[0], goal[1], "xg", label="goal")
    # plt.axis("equal")
    # plt.axis([0, 120, -105, 15])
    # ss=0
    # s0s=0
    # for obs in obstacle_list:
    #     if len(obs) == 3:
    #         plot_obs(obs[0], obs[1], obs[2],"y",ss)
    #         ss+=1
    #     elif len(obs) == 4:
    #         plot_obs_rec(obs[0], obs[1], obs[2], obs[3],"b",s0s)
    #         s0s+=1
    # plt.xlabel("x/m")
    # plt.ylabel("y/m")
    # plt.legend()
    # plt.show()

    # path, dwa, x, goal, obstacles, config, rrt_path = rrt_dwa(start, goal, obstacle_list, True)

    # if path is not None:
    #     print("Path found and animation saved.")
    # else:
    #     print("No path found")
