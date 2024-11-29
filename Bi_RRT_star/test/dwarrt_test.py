import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from single_planning.DWA import DWA
from single_planning.Bi_RRT import RRT_plan  # 假设你有一个RRT_plan函数


def plot_obstacle(ax, x, y, radius):
    circle = plt.Circle((x, y), radius, color='r')
    ax.add_patch(circle)


def main():
    global x, obstacles, path, dwa_mode, rrt_path  # Declare global variables

    # DWA configuration
    config = {
        'max_speed': 1.0,
        'min_speed': -0.5,
        'max_yawrate': 40.0 * np.pi / 180.0,
        'max_accel': 0.2,
        'max_dyawrate': 40.0 * np.pi / 180.0,
        'v_reso': 0.01,
        'yawrate_reso': 0.1 * np.pi / 180.0,
        'dt': 0.1,
        'predict_time': 3.0,
        'to_goal_cost_gain': 1.0,
        'speed_cost_gain': 1.0,
        'robot_radius': 1.0
    }

    dwa = DWA(config)

    # Initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([10.0, 10.0])
    obstacles = [[5.0, 5.0]]
    path = [x[:2].copy()]  # Initialize path with the starting position
    dwa_mode = True  # Start in DWA mode
    rrt_path = RRT_plan([0.0, 0.0], goal.tolist(), obstacles)  # Initial RRT path

    fig, ax = plt.subplots()
    ax.plot(goal[0], goal[1], "xk")
    plot_obstacle(ax, obstacles[0][0], obstacles[0][1], config['robot_radius'])
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)

    def update(frame):
        global x, obstacles, path, dwa_mode, rrt_path  # Access global variables

        if dwa_mode:
            u, trajectory = dwa.plan(x, goal, obstacles)
            x = dwa.motion(x, u)
            path.append(x[:2].copy())  # Append current position to path

            # Check if we should exit DWA mode
            if not obstacles_in_front(x, obstacles):
                dwa_mode = False
                # Find the nearest point in the original path
                nearest_point = find_nearest_point(x, rrt_path)
                goal[:] = nearest_point  # Set new goal to the nearest point

        else:
            # Follow the RRT path
            if np.linalg.norm(x[:2] - goal) < 0.5:  # If close to the goal, move to the next point
                if len(rrt_path) > 1:
                    rrt_path.pop(0)
                    goal[:] = rrt_path[0]
                else:
                    print("Reached the final goal")
                    return

            u, trajectory = dwa.plan(x, goal, obstacles)
            x = dwa.motion(x, u)
            path.append(x[:2].copy())  # Append current position to path

        ax.cla()
        ax.plot(goal[0], goal[1], "xk")
        for obs in obstacles:
            plot_obstacle(ax, obs[0], obs[1], config['robot_radius'])
        ax.plot(x[0], x[1], "ob")
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 11)

        # Plot the path
        for p in path[::4]:  # Print every third point
            ax.plot(p[0], p[1], ".g")

    ani = animation.FuncAnimation(fig, update, frames=100, repeat=False)
    ani.save('dwa_animation.gif', writer='imagemagick')
    print("path:", path)


def obstacles_in_front(x, obstacles):
    # Check if there are obstacles in front of the robot
    for obs in obstacles:
        if np.linalg.norm(x[:2] - obs[:2]) < 5.0:  # Example threshold
            return True
    return False


def find_nearest_point(x, path):
    # Find the nearest point in the path to the current position
    min_dist = float('inf')
    nearest_point = path[0]
    for point in path:
        dist = np.linalg.norm(x[:2] - point)
        if dist < min_dist:
            min_dist = dist
            nearest_point = point
    return nearest_point


if __name__ == "__main__":
    main()
