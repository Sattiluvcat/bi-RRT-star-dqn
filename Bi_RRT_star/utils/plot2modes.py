import matplotlib.pyplot as plt

def plot_obstacle(ax, x, y, radius=None, width=None, height=None):
    if radius is not None:
        circle = plt.Circle((x, y), radius, color='r')
        ax.add_patch(circle)
    elif width is not None and height is not None:
        rectangle = plt.Rectangle((x, y), width, height, color='r')
        ax.add_patch(rectangle)
