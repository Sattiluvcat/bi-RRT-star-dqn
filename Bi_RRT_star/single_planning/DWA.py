import numpy as np

from single_planning.Bi_RRT import check_collision, calc_p2l_xianduan_dis
from utils.node import Node


# 计算与障碍物的最近距离
def calc_obstacle_cost(trajectory, obstacles):
    min_dis = float('inf')
    # 根据config参数进行调整，当前取前1.5s的轨迹进行判断
    for i in range(int(len(trajectory) / 10*3) - 1):
        start_pos = Node(trajectory[i, 0], trajectory[i, 1])
        end_pos = Node(trajectory[i + 1, 0], trajectory[i + 1, 1])
        for obs in obstacles:
            dis = calc_p2l_xianduan_dis(start_pos, end_pos, (obs[0], obs[1]))
            if dis < obs[2]:  # obs[2] is the radius of the obstacle
                dis = float('inf')  # the path intersects with the obstacle
                return dis
            else:
                dis -= obs[2]  # the path is outside the obstacle
            if dis < min_dis:
                min_dis = dis
    return 0
    # if check_collision(start_pos, end_pos, obstacles):
    #     return float("inf")  # collision
    # return 0  # no collision


def calc_to_goal_cost(trajectory, goal):
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    return np.hypot(dx, dy)


class DWA:
    def __init__(self, config):
        self.config = config

    def plan(self, x, goal, obstacles):
        # Dynamic Window [v_min, v_max, omega_min, omega_max]
        dw = self.calc_dynamic_window(x)
        u, trajectory = self.calc_control_and_trajectory(x, dw, goal, obstacles)
        return u, trajectory

    def calc_dynamic_window(self, x):
        # Dynamic window from robot specification
        Vs = [self.config['min_speed'], self.config['max_speed'],
              -self.config['max_yawrate'], self.config['max_yawrate']]

        # Dynamic window from motion model
        Vd = [x[3] - self.config['max_accel'] * self.config['dt'],
              x[3] + self.config['max_accel'] * self.config['dt'],
              x[4] - self.config['max_dyawrate'] * self.config['dt'],
              x[4] + self.config['max_dyawrate'] * self.config['dt']]

        # [v_min, v_max, omega_min, omega_max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw

    def calc_control_and_trajectory(self, x, dw, goal, obstacles):
        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array(x)

        # evaluate all trajectory with sampled input in dynamic window
        for v in np.arange(dw[0], dw[1], self.config['v_reso']):
            for y in np.arange(dw[2], dw[3], self.config['yawrate_reso']):
                trajectory = self.calc_trajectory(x_init, v, y)
                to_goal_cost = self.config['to_goal_cost_gain'] * calc_to_goal_cost(trajectory, goal)
                speed_cost = self.config['speed_cost_gain'] * (self.config['max_speed'] - trajectory[-1, 3])
                ob_cost = calc_obstacle_cost(trajectory, obstacles)
                # TODO 向量场的cost——vf时设置
                field_cost=0

                # cost最小的轨迹
                final_cost = to_goal_cost + speed_cost + ob_cost

                if min_cost > final_cost != float('inf'):
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory

        return best_u, best_trajectory

    def calc_trajectory(self, x_init, v, y):
        trajectory = np.array(x_init)
        x = np.array(x_init)
        time = 0
        while time <= self.config['predict_time']:
            x = self.motion(x, [v, y])
            trajectory = np.vstack((trajectory, x))
            time += self.config['dt']
        return trajectory

    def motion(self, x, u):
        x[2] += u[1] * self.config['dt']
        x[0] += u[0] * np.cos(x[2]) * self.config['dt']
        x[1] += u[0] * np.sin(x[2]) * self.config['dt']
        x[3] = u[0]
        x[4] = u[1]
        return x
