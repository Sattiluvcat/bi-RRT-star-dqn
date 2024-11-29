import matplotlib.pyplot as plt
from matplotlib import animation


def plot_obstacle(ax, x, y, radius):
    circle = plt.Circle((x, y), radius, color='r')
    ax.add_patch(circle)


def plot_dwa_path(fig, ax, dwa, x, goal, obstacles, path, config):
    def update(frame):
        global x, obstacles, path  # Access global variables
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
        print("path:", path)
        for p in path[::4]:
            print("p:", p)
            ax.plot(p[0], p[1], ".g")

        # Dynamically add obstacle
        if frame == 50:
            new_obstacle = [7.0, 7.0]
            obstacles.append(new_obstacle)

    ani = animation.FuncAnimation(fig, update, frames=30, repeat=False)
    ani.save('dwa_animation.gif', writer='imagemagick')
    print("path:", path)


def plot_rrt_path(fig, ax, rrt_path, obstacles, config):
    ax.cla()
    for obs in obstacles:
        plot_obstacle(ax, obs[0], obs[1], config['robot_radius'])
    for point in rrt_path:
        ax.plot(point[0], point[1], "xk")
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    plt.draw()


def plot_path(fig, ax, mode, dwa, x, goal, obstacles, path, config, rrt_path):
    if mode == 'dwa':
        plot_dwa_path(fig, ax, dwa, x, goal, obstacles, path, config)
    elif mode == 'rrt':
        plot_rrt_path(fig, ax, rrt_path, obstacles, config)
