# 导入用到的库
import math
import random
import time

import matplotlib.pyplot as plt
from shapely.geometry.geo import box
from shapely.geometry.linestring import LineString

from utils.node import Node


# 计算两点间的距离
def calc_p2p_dis(point1, point2):
    if isinstance(point1, Node):
        x1, y1 = point1.x, point1.y
    else:  # assume point1 is a list or tuple
        x1, y1 = point1

    if isinstance(point2, Node):
        x2, y2 = point2.x, point2.y
    else:  # assume point2 is a list or tuple
        x2, y2 = point2

    dx = x2 - x1
    dy = y2 - y1
    d = math.sqrt(dx ** 2 + dy ** 2)
    return d


# 计算点到线的距离
def calc_p2l_dis(line_point1, line_point2, point):  # 到两点确定的直线的距离 非到线段的最小距离
    if isinstance(line_point1, Node):
        x1, y1 = line_point1.x, line_point1.y
    else:
        x1, y1 = line_point1

    if isinstance(line_point2, Node):
        x2, y2 = line_point2.x, line_point2.y
    else:
        x2, y2 = line_point2
    if isinstance(point, Node):
        x3, y3 = point.x, point.y
    else:
        x3, y3 = point
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    if A ** 2 + B ** 2 == 0:
        return 0
    dis = abs(A * x3 + B * y3 + C) / math.sqrt(A ** 2 + B ** 2)
    return dis


#定义到线段的距离
def calc_p2l_xianduan_dis(line_point1, line_point2, point):
    if isinstance(line_point1, Node):
        x1, y1 = line_point1.x, line_point1.y
    else:
        x1, y1 = line_point1
    if isinstance(line_point2, Node):
        x2, y2 = line_point2.x, line_point2.y
    else:
        x2, y2 = line_point2
    if isinstance(point, Node):
        x3, y3 = point.x, point.y
    else:
        x3, y3 = point
    px = x2 - x1
    py = y2 - y1
    something = px * px + py * py
    u = ((x3 - x1) * px + (y3 - y1) * py) / float(something)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    x = x1 + u * px
    y = y1 + u * py
    dx = x - x3
    dy = y - y3
    return math.sqrt(dx * dx + dy * dy)


# 计算角BAC的角度
def calc_triangle_deg(pointA, pointB, pointC):
    if isinstance(pointA, Node):
        pointA = [pointA.x, pointA.y]
    else:
        pointA = list(pointA)
    if isinstance(pointB, Node):
        pointB = [pointB.x, pointB.y]
    else:
        pointB = list(pointB)
    if isinstance(pointC, Node):
        pointC = [pointC.x, pointC.y]
    else:
        pointC = list(pointC)

    AB = [pointB[i] - pointA[i] for i in range(len(pointA))]
    AC = [pointC[i] - pointA[i] for i in range(len(pointA))]
    dot_product = sum(AB[i] * AC[i] for i in range(len(pointA)))
    mod_AB = math.sqrt(sum(AB[i] ** 2 for i in range(len(pointA))))
    mod_AC = math.sqrt(sum(AC[i] ** 2 for i in range(len(pointA))))
    if mod_AB == 0 or mod_AC == 0:
        return 0
    cos_angle = dot_product / (mod_AB * mod_AC)
    cos_angle = max(-1, min(cos_angle, 1))
    angle_in_radians = math.acos(cos_angle)
    angle_in_degrees = math.degrees(angle_in_radians)
    return angle_in_degrees


# 在规定区域内生成随机点
def get_random_node(x_min, x_max, y_min, y_max, goal_point=None):
    if goal_point is not None and random.random() <= 0.2:
        x = goal_point[0]
        y = goal_point[1]
    else:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
    rnd_node = Node(x, y)
    return rnd_node


# 寻找距离随机点最近的节点在节点列表中的位置
def get_nearest_node_index(node_list, rnd_node):
    dlist = []
    for node in node_list:
        dis = calc_p2p_dis(node, rnd_node)
        dlist.append(dis)
    minind = dlist.index(min(dlist))
    #    print(minind)
    return minind


# 根据随机点和距离随机点最近的点生成新的点
def generate_new_node(nearest_node, random_node, extend_length):
    new_node = Node(nearest_node.x, nearest_node.y)
    d = math.sqrt((random_node.x - nearest_node.x) ** 2 + (random_node.y - nearest_node.y) ** 2)
    if extend_length > d:
        extend_length = d
    dx = random_node.x - nearest_node.x
    dy = random_node.y - nearest_node.y
    if dx * dx + dy * dy == 0:
        return random_node
    new_node.x += extend_length / math.sqrt(dx * dx + dy * dy) * dx
    new_node.y += extend_length / math.sqrt(dx * dx + dy * dy) * dy
    new_node.parent = nearest_node  # ??
    return new_node


# 生成路径
def generate_final_course(goal_node, node_list):  # 原路径中多加入一点
    path_reverse = [[goal_node.x, goal_node.y]]
    node = node_list[-1]
    while node.parent != None:
        path_reverse.append([node.x, node.y])
        node = node.parent
    path_reverse.append([node.x, node.y])
    path = []
    for i in range(len(path_reverse)):
        path.append(path_reverse[-(i + 1)])
    return path


# 剪枝
def prune_path(path, obs_list):
    pruned_path = [path[0]]
    i = 0
    while i < len(path) - 1:
        found = False
        for j in range(len(path) - 1, i, -1):
            if not check_collision(path[i], path[j], obs_list):
                pruned_path.append(path[j])
                i = j
                found = True
                break
        if not found:
            # 确保路径前进
            i += 1
    if pruned_path[-1] != path[-1]:
        pruned_path.append(path[-1])
    return pruned_path


def RRT_plan(start_xy, goal_xy,
             obslis_xy):
    print("起始点：")
    print(start_xy)
    print(goal_xy)
    x_min = min(start_xy[0], goal_xy[0]) - 10
    x_max = max(start_xy[0], goal_xy[0]) + 10
    y_min = min(start_xy[1], goal_xy[1]) - 10
    y_max = max(start_xy[1], goal_xy[1]) + 10
    print(x_min, x_max, y_min, y_max)
    start_point = start_xy
    goal_point = goal_xy
    obs_list = obslis_xy
    extend_length = 5
    mini_degree = 90
    max_iter = 10000
    start_node = Node(start_point[0], start_point[1])
    goal_node = Node(goal_point[0], goal_point[1])
    node_list1 = [start_node]  # 从起点开始的树
    node_list2 = [goal_node]  # 从终点开始的树
    path = None
    if not check_collision(start_node, goal_node, obs_list):
        path = [start_point, goal_point]
        return path
    else:
        for i in range(max_iter):
            rnd_nd = get_random_node(x_min, x_max, y_min, y_max, goal_point)
            near_index1 = get_nearest_node_index(node_list1, rnd_nd)
            near_index2 = get_nearest_node_index(node_list2, rnd_nd)
            new_nd1 = generate_new_node(node_list1[near_index1], rnd_nd, extend_length)
            new_nd2 = generate_new_node(node_list2[near_index2], rnd_nd, extend_length)
            # 转角约束
            if node_list1[near_index1 - 1] is not None and node_list2[near_index2 - 1] is not None:
                degree1 = calc_triangle_deg(node_list1[near_index1], rnd_nd, node_list1[near_index1 - 1])
                degree2 = calc_triangle_deg(node_list2[near_index2], rnd_nd, node_list2[near_index2 - 1])
                if degree1 < mini_degree and degree1 != 0 and degree2 < mini_degree and degree2 != 0:
                    continue
            if new_nd1 is not None and check_collision(new_nd1, node_list1[near_index1], obs_list) == False:
                new_nd1.parent = node_list1[near_index1]
                node_list1.append(new_nd1)
                plt.plot(new_nd1.x, new_nd1.y, "xg")
                plt.plot([new_nd1.parent.x, new_nd1.x], [new_nd1.parent.y, new_nd1.y], 'g')
            if new_nd2 is not None and check_collision(new_nd2, node_list2[near_index2], obs_list) == False:
                new_nd2.parent = node_list2[near_index2]
                node_list2.append(new_nd2)
                plt.plot(new_nd2.x, new_nd2.y, "xb")
                plt.plot([new_nd2.parent.x, new_nd2.x], [new_nd2.parent.y, new_nd2.y], 'b')
            plt.axis("equal")
            plt.axis([0.0, 260.0, -200.0, 10.0])
            for node1 in node_list1:
                node2 = new_nd2
                if calc_p2p_dis(node1, node2) <= extend_length and \
                        check_collision(node1, node2, obs_list) == False:
                    # 生成从起点到相交点的路径
                    path1 = []
                    node = node1
                    while node is not None:
                        path1.append([node.x, node.y])
                        node = node.parent
                    path1.reverse()  # 反转路径，使其从起点开始
                    # 生成从终点到相交点的路径
                    path2 = []
                    node = node2
                    while node is not None:
                        path2.append([node.x, node.y])
                        node = node.parent
                    # 合并两条路径
                    path = path1 + path2
                    # return path
                    return prune_path(path, obs_list)
            for node2 in node_list2:
                node1 = new_nd1
                if calc_p2p_dis(node1, node2) <= extend_length and \
                        check_collision(node1, node2, obs_list) == False:
                    # 生成从起点到相交点的路径
                    path1 = []
                    node = node1
                    while node is not None:
                        path1.append([node.x, node.y])
                        node = node.parent
                    path1.reverse()  # 反转路径，使其从起点开始
                    # 生成从终点到相交点的路径
                    path2 = []
                    node = node2
                    while node is not None:
                        path2.append([node.x, node.y])
                        node = node.parent
                    # 合并两条路径
                    path = path1 + path2
                    # return path
                    return prune_path(path, obs_list)
        return None


#计算路径总长
def calc_path_length(path):
    total_length = 0
    for i in range(len(path) - 1):
        total_length += calc_p2p_dis(path[i], path[i + 1])
    return total_length


def check_collision(node1, node2, obstacleList):
    if isinstance(node1, Node):
        node1 = [node1.x, node1.y]
    if isinstance(node2, Node):
        node2 = [node2.x, node2.y]

    for obstacle in obstacleList:
        if len(obstacle) == 3:  # 圆形障碍物
            ox, oy, size = obstacle
            round = Node(ox, oy)
            size_new = size
            if calc_p2p_dis(Node(node1[0], node1[1]), round) <= size_new:
                return True
            if calc_p2p_dis(Node(node2[0], node2[1]), round) <= size_new:
                return True
            if calc_p2l_dis(Node(node1[0], node1[1]), Node(node2[0], node2[1]), round) <= size_new and \
                    calc_triangle_deg(Node(node1[0], node1[1]), round, Node(node2[0], node2[1])) <= 90 and \
                    calc_triangle_deg(Node(node2[0], node2[1]), round, Node(node1[0], node1[1])) <= 90:
                return True
        elif len(obstacle) == 4:  # 矩形障碍物
            rect_shape = box(obstacle[0], obstacle[1], obstacle[2], obstacle[3])
            line = LineString([(node1[0], node1[1]), (node2[0], node2[1])])
            if line.intersects(rect_shape):
                return True
    return False


# 重新定义障碍物格式——矩形保持左下到右上
def re_obs(obs_list):
    for obs in obs_list:
        if len(obs) == 4:
            if obs[0] > obs[2]:
                tmp = obs[0]
                obs[0] = obs[2]
                obs[2] = tmp
                tmp = obs[1]
                obs[1] = obs[3]
                obs[3] = tmp
    return obs_list


# 画圆
def plot_obs(x, y, size, color="-b"):  # pragma: no cover
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * math.cos(math.radians(d)) for d in deg]
    yl = [y + size * math.sin(math.radians(d)) for d in deg]
    plt.plot(xl, yl, color)

# 画方
def plot_obs_rec(x1, y1, x2, y2, color="-b"):
    xl = [x1, x2, x2, x1, x1]
    yl = [y1, y1, y2, y2, y1]
    plt.plot(xl, yl, color)


def path_score(path, all_time, obs_list):
    print("运行时间：", all_time, "s")
    print("路径总长：", calc_path_length(path), "m")
    length = calc_path_length(path)
    total_angle = 0
    if len(path) >= 3:
        for i in range(len(path) - 2):
            total_angle += abs(180 - calc_triangle_deg(path[i + 1], path[i], path[i + 2]))
    else:
        total_angle = 0
    print("累积转角：", total_angle, "°")
    min_dis = float('inf')
    obstacle_list = obs_list
    # obstacle_list = [[latlon_to_xy(obstacle[1], obstacle[0])[0],
    #                   latlon_to_xy(obstacle[1], obstacle[0])[1], obstacle[2]] for obstacle in obs_list]
    for i in range(len(path) - 1):
        for obs in obstacle_list:
            dis = calc_p2l_xianduan_dis(path[i], path[i + 1], (obs[0], obs[1]))
            if dis < obs[2]:  # obs[2] is the radius of the obstacle
                dis = 0  # the path intersects with the obstacle
            else:
                dis -= obs[2]  # the path is outside the obstacle
            if dis < min_dis:
                min_dis = dis
    print("最近距离：", min_dis, "m")
    score = 35 * (1 - all_time / 2) + 35 * (1 - (length - 300) / 100) + 20 * (1 - total_angle / 180) + 10 * (
            min_dis - 10) / 0.5
    print("路径评分：", score)
    return score