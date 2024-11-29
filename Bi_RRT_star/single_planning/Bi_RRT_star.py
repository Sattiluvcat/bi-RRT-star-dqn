# 导入用到的库
from single_planning.Bi_RRT import *

from utils.node import Node


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


def Bi_RRT_star_plan(start_xy, goal_xy,
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
            # 重布线与重写操作效果检验
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