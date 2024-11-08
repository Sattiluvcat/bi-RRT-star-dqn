# 导入用到的库
import math
import random
import time

import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely.geometry.geo import box
from shapely.geometry.linestring import LineString

import Bi_RRT_star_main

# from matplotlib import pyplot as plt
# 定义RRT的节点类
class Node:
    # 初始化节点
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost=0.0


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
    if not isinstance(pointA, list):
        pointA = [pointA.x, pointA.y]
    if not isinstance(pointB, list):
        pointB = [pointB.x, pointB.y]
    if not isinstance(pointC, list):
        pointC = [pointC.x, pointC.y]
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


# 检查点1和点2连接成的线段是否与障碍物碰撞，若碰撞则返回True，反则反之
def check_collision(node1, node2, obstacleList):
    for obstacle in obstacleList:
        if len(obstacle) == 3:  # 圆形障碍物
            ox, oy, size = obstacle
            round = Node(ox, oy)
            size_new = size
            if calc_p2p_dis(node1, round) <= size_new:
                return True
            if calc_p2p_dis(node2, round) <= size_new:
                return True
            if calc_p2l_dis(node1, node2, round) <= size_new and calc_triangle_deg(node1, round, node2) <= 90 and \
                    calc_triangle_deg(node2, round, node1) <= 90:
                return True
        elif len(obstacle) == 4:  # 矩形障碍物
            rect_shape = box(obstacle[0], obstacle[1], obstacle[2], obstacle[3])
            if not isinstance(node1, list):
                line = LineString([(node1.x, node1.y), (node2.x, node2.y)])
            else:
                line = LineString([(node1[0], node1[1]), (node2[0], node2[1])])
            if line.intersects(rect_shape):
                return True
    return False


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


# transformer=Transformer.from_crs("epsg:4326", "epsg:3857")
# def latlon_to_xy(lat, lon):
#     x, y = transformer.transform(lat, lon)
#     x=x-13308770
#     y=y-4219000
#     return x, y

# def xy_to_latlon(x, y):
#     x=x+13308770
#     y=y+4219000
#     lon, lat = transformer.transform(x,y,direction='INVERSE')
#     return lat, lon

# def rewrite_index(node_new, node_list, obstacle_list):
#     r = 10  # 定义搜索范围 应大于步长（包括原parent节点）
#     dislist = []
#     dlist = []
#     # print("rewrite index:"+node_new.x)
#     i = 0
#     for node in node_list:
#         if calc_p2p_dis(node_new, node) < r and not check_collision(node_new, node, obstacle_list):
#             potential_dis = node.cost + calc_p2p_dis(node_new, node)
#             dislist.append(potential_dis)
#             dlist.append(i)
#         i = i + 1
#     min_node = dlist[dislist.index(min(dislist))]
#     return min_node

def rewrite_index(node_new, node_list, obstacle_list):
    r = 8  # Define search radius
    min_cost = float('inf')
    min_node_index = None

    for i, node in enumerate(node_list):
        if calc_p2p_dis(node_new, node) < r and not check_collision(node_new, node, obstacle_list):
            potential_cost = node.cost + calc_p2p_dis(node_new, node)
            if potential_cost < min_cost:
                min_cost = potential_cost
                min_node_index = i

    return min_node_index

#重布线——所有可更改parent的节点都更改/第一个找到的更改
# def rewire(node_new, node_list, obstacle_list):
#     r = 20  # 查找区域
#     i = 0
#     for node in node_list:
#         if calc_p2p_dis(node, node_new.parent) != 0 and calc_p2p_dis(node_new, node) < r:  # 排除新节点自己的parent节点
#             dis_now = node_new.cost + calc_p2p_dis(node, node_new)
#             if dis_now < node.cost:
#                 node.parent = node_new  # 重选新节点
#                 node.cost = dis_now
#                 # break                 #若取消注释，则只有第一个满足该条件的节点的parent会变为node_new

def rewire(node_new, node_list, obstacle_list):
    r = 30  # Define search radius

    for node in node_list:
        if node != node_new.parent and calc_p2p_dis(node_new, node) < r:
            potential_cost = node_new.cost + calc_p2p_dis(node, node_new)
            # 下面这个限定到底要不要👇
            # if potential_cost < node.cost and check_collision(node, node_new, obstacle_list) is False:
            if potential_cost < node.cost:
                node.parent = node_new
                node.cost = potential_cost

def RRT_star_plan(start_xy, goal_xy,
                  obslis_xy):
    # point:[x,y], boundary:[x_min,x_max,y_min,y_max], obs_list:[[x1,y1,r1],...]
    # start_xy=latlon_to_xy(start_point[1],start_point[0])
    # goal_xy=latlon_to_xy(goal_point[1],goal_point[0])
    print("起始点：")
    print(start_xy)
    print(goal_xy)
    x_min = min(start_xy[0], goal_xy[0]) - 10
    x_max = max(start_xy[0], goal_xy[0]) + 10
    y_min = min(start_xy[1], goal_xy[1]) - 10
    y_max = max(start_xy[1], goal_xy[1]) + 10
    print(x_min, x_max, y_min, y_max)
    # obslis_xy=[[latlon_to_xy(obstacle[1], obstacle[0])[0],
    #             latlon_to_xy(obstacle[1], obstacle[0])[1], obstacle[2]+10] for obstacle in obs_list]
    start_point = start_xy
    goal_point = goal_xy
    obs_list = obslis_xy
    extend_length = 5
    # mini_degree = 90
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
            # print(i)
            rnd_nd = get_random_node(x_min, x_max, y_min, y_max, goal_point)
            near_index1 = get_nearest_node_index(node_list1, rnd_nd)
            near_index2 = get_nearest_node_index(node_list2, rnd_nd)
            new_nd1 = generate_new_node(node_list1[near_index1], rnd_nd, extend_length)
            new_nd2 = generate_new_node(node_list2[near_index2], rnd_nd, extend_length)
            # if node_list1[near_index1 - 1] != None and node_list2[near_index2 - 1] != None:
            #     degree1 = calc_triangle_deg(node_list1[near_index1], rnd_nd, node_list1[near_index1 - 1])
            #     degree2 = calc_triangle_deg(node_list2[near_index2], rnd_nd, node_list2[near_index2 - 1])
            #     if degree1 < mini_degree and degree1 != 0 and degree2 < mini_degree and degree2 != 0:
            #         continue
            if new_nd1 is not None and check_collision(new_nd1, node_list1[near_index1], obs_list) == False:
                parent_index = rewrite_index(new_nd1, node_list1, obs_list)
                new_nd1.parent = node_list1[parent_index]
                # node_list1.pop(near_index1)
                node_list1.append(new_nd1)
                new_nd1.cost = new_nd1.parent.cost + calc_p2p_dis(new_nd1, new_nd1.parent)
                rewire(new_nd1, node_list1, obs_list)

                plt.plot(new_nd1.x, new_nd1.y, "xg")
                plt.plot([new_nd1.parent.x, new_nd1.x], [new_nd1.parent.y, new_nd1.y], 'g')
            if new_nd2 is not None and check_collision(new_nd2, node_list2[near_index2], obs_list) == False:
                parent_index = rewrite_index(new_nd2, node_list2, obs_list)
                new_nd2.parent = node_list2[parent_index]
                # node_list2 = node_list2[:(parent_index + 1)]
                node_list2.append(new_nd2)
                new_nd2.cost = new_nd2.parent.cost + calc_p2p_dis(new_nd2, new_nd2.parent)
                rewire(new_nd2, node_list2, obs_list)

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


# 画圆
def plot_obs(x, y, size, color="-b"):  # pragma: no cover
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * math.cos(math.radians(d)) for d in deg]
    yl = [y + size * math.sin(math.radians(d)) for d in deg]
    plt.plot(xl, yl, color)

def plot_obs_rec(x1,y1,x2,y2, color="-b"):
    xl=[x1,x2,x2,x1,x1]
    yl=[y1,y1,y2,y2,y1]
    plt.plot(xl,yl,color)

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
    # obstacle_list = [[latlon_to_xy(obstacle[1], obstacle[0])[0],
    #                   latlon_to_xy(obstacle[1], obstacle[0])[1], obstacle[2]] for obstacle in obs_list]
    obstacle_list=obs_list
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

# 重新定义障碍物格式——矩形保持左下到右上
def re_obs(obs_list):
    for obs in obs_list:
        if len(obs) ==4:
            if obs[0]>obs[2]:
                tmp=obs[0]
                obs[0]=obs[2]
                obs[2]=tmp
                tmp=obs[1]
                obs[1]=obs[3]
                obs[3]=tmp
    return obs_list

if __name__ == "__main__":
    start_time = time.time()

    # 画图
    start = [8.77650817669928, 4.951398874633014]
    goal = [249.09851929917932, -195.2040752498433]
    obstacle_list = [[33.77685192972422, -52.90446031652391, 7.5], [46.19124796241522, -77.85628066305071, 9.0],
                     [80.34613360464573, -43.54909089393914, 7.5], [111.38825733587146, -74.74188929889351, 7.5],
                     [80.16235236078501, 9.456780110485852, 4.5], [139.31754435040057, -12.368450773879886, 4.5],
                     [207.6151233687997, 3.4003808852285147, 15.0], [111.59657261520624, -127.13194767106324, 15.0],
                     [307.5552379246801, -6.496737029403448, 15.0], [182.18576977215707, -97.71181975770742, 15.0],
                     [232.431648472324, -58.93625687714666, 220,-70], [76.79213218018413, -161.22955951932818, 7.5],
                     [168.2150300759822, -149.5354239968583, 7.5], [265.8879858329892, -127.0088061131537, 7.5],
                     [325.93784911744297, -71.36904122401029, 7.5], [284.0254806391895, -45.27248460613191, 4.5]]

    obstacle_list=re_obs(obstacle_list)

    path = RRT_star_plan(start, goal, obstacle_list)
    print(path)

    # goal = latlon_to_xy(goal[1], goal[0])
    obstacle = obstacle_list
    plt.plot(start[0], start[1], "xk")
    plt.plot(goal[0], goal[1], "xk")
    plt.axis("equal")
    plt.axis([0, 260, -200, 10])

    for obs in obstacle:
        if len(obs)==3:
            plot_obs(obs[0], obs[1], obs[2])
        elif len(obs)==4:
            plot_obs_rec(obs[0], obs[1], obs[2], obs[3])
    # for [x, y, size] in obstacle:
    #     plot_obs(x, y, size)

    # 画图
    plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r')
    # plt.pause(0.01)
    plt.show()
    end_time = time.time()
    all_time = end_time - start_time
    path_score(path, all_time, obstacle_list)
