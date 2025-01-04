# 类似Bi_RRT & Bi_RRT*子类，考虑流场情况下的rrt，新增生成新点、计算上游系数并选择最小cost路径的函数
import numpy as np

from single_planning.Bi_RRT_star import *
from utils.node import Node


# 结合向量场方向得到新节点
def vf_generate_new_node(nearest_node, random_node, extend_length, vector_field):
    # 获取最近节点到随机点的方向向量
    direction_to_random = np.array([random_node.x - nearest_node.x, random_node.y - nearest_node.y])
    # 不该出现0的情况（原来是回旋镖）
    if np.linalg.norm(direction_to_random)!=0:
        direction_to_random /= np.linalg.norm(direction_to_random)
    print("nearest_node:", nearest_node.x, nearest_node.y)

    # 获取最近节点处的向量场方向
    u, v = get_vector_field(nearest_node.x, nearest_node.y, vector_field)
    direction_vector_field = np.array([u, v])

    # TODO 方向选取的优化
    # 两个方向向量的加权平均方向
    average_direction_vector = (direction_to_random*4 + direction_vector_field) / 5

    # 归一化平均方向向量
    average_direction_vector /= np.linalg.norm(average_direction_vector)

    # 根据这个平均方向生成新节点
    new_x = nearest_node.x + extend_length * average_direction_vector[0]
    new_y = nearest_node.y + extend_length * average_direction_vector[1]

    new_node = Node(new_x, new_y)
    new_node.parent = nearest_node

    return new_node


# 得到该位置向量场方向——插值
def get_vector_field(x, y, vector_field):
    X, Y, U, V = vector_field
    x1 = int(np.floor(x))+20
    # 向量场的坐标系是从-220到20,U V的坐标系是从0开始
    y1 = int(np.floor(y))+220
    x2 = x1 + 1
    y2 = y1 + 1

    # u = U[y1, x1]*(x2-x) * (y2-y) + U[y1, x2]*(x-x1) * (y2-y) + U[y2, x1]*(x2-x) * (y-y1) + U[y2, x2]*(x-x1) * (y-y1)
    u = U[y1, x1]
    v = V[y1, x1]
    return u, v


# 计算路径中每一步的上游系数结果（rrt-dwa方法）
def upstream_criterion(path, vector_field):
    total_difference = 0
    for i in range(1, len(path)):
        # 当前位置的向量场
        u, v = get_vector_field(path[i][0], path[i][1], vector_field)
        vector_field_magnitude = np.sqrt(u**2 + v**2)
        # 本来已经归一化了 但是计算精度可能不准确 此处二加工
        direction_vector_field = np.array([u, v]) / vector_field_magnitude

        # 当前速度方向——路径求导
        direction_path = np.gradient(np.array(path), axis=0)[i]
        path_magnitude = np.linalg.norm(direction_path)
        # 归一化速度方向 --> 需要，因为向量场本身已经归一化 --> 应用不等式时两者模长相等，均为1
        direction_path /= path_magnitude

        # Cauchy-Schwarz 不等式: |a · b| <= ||a|| * ||b||
        dot_product = np.dot(direction_path, direction_vector_field)
        # 👆归一化后直接取 1 即可
        total_difference += 1 * vector_field_magnitude - dot_product
    return total_difference


# 选择上游系数最小的路径
def choose_lowest_cost(paths, vector_field):
    min_cost = float('inf')
    best_path = None
    for path in paths:
        cost = upstream_criterion(path, vector_field)
        if cost < min_cost:
            min_cost = cost
            best_path = path
    return best_path


# 考虑流场的剪枝
def vf_prune_path(path, obs_list, vector_field):
    pruned_path = [path[0]]
    i = 0

    while i < len(path) - 1:
        found = False
        for j in range(len(path) - 1, i, -1):
            # TODO 转角约束
            if not check_collision(path[i], path[j], obs_list):
                # 现有优化路径+现在考虑的路径不剪枝形式
                candidate_path = pruned_path + path[i:]

                # 计算路径评分——跳过 i 到 j 中间的路径
                score_start_to_current = path_score(pruned_path + path[j:], vector_field)
                # 计算路径评分——已优化路径
                score_current_to_previous = path_score(candidate_path, vector_field)

                # Compare the scores
                if score_start_to_current < score_current_to_previous:
                    pruned_path.append(path[j])
                    i = j
                    found = True
                    break
        if not found:
            i += 1
    if pruned_path[-1] != path[-1]:
        pruned_path.append(path[-1])
    return pruned_path


# 剪枝中的评分函数
def path_score(path, vector_field):
    total_difference = 0
    total_angle = 0
    # TODO 评分细则
    for i in range(len(path) - 1):  # 最大值是 len(path) - 2
        u, v = get_vector_field(path[i][0], path[i][1], vector_field)
        direction_vector_field = np.arctan2(v, u)
        direction_path = np.arctan2(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0])
        total_difference += abs(direction_path - direction_vector_field)

        if i < len(path) - 2:
            angle = abs(np.arctan2(path[i + 2][1] - path[i + 1][1], path[i + 2][0] - path[i + 1][0]) - direction_path)
            total_angle += angle

    return total_difference + total_angle

# TODO 现有效果一般
def VF_Bi_RRT_star_plan(start_xy, goal_xy, obslis_xy, vector_field):
    x_min = min(start_xy[0], goal_xy[0]) - 10
    x_max = max(start_xy[0], goal_xy[0]) + 10
    y_min = min(start_xy[1], goal_xy[1]) - 10
    y_max = max(start_xy[1], goal_xy[1]) + 10
    start_point = start_xy
    goal_point = goal_xy
    obs_list = obslis_xy
    extend_length = 5
    max_iter = 10000
    # 路径总数
    path_num = 5

    start_node = Node(start_point[0], start_point[1])
    goal_node = Node(goal_point[0], goal_point[1])
    node_list1 = [start_node]
    node_list2 = [goal_node]
    paths = []

    # 画向量场
    X, Y, U, V = vector_field
    plt.quiver(X[::10, ::10], Y[::10, ::10], U[::10, ::10], V[::10, ::10], color='b')

    # 记录路径是否找到，找到后跳出for循环
    path_found = False

    while path_num > 0:
        for i in range(max_iter):
            rnd_nd1 = get_random_node(x_min, x_max, y_min, y_max, goal_point)
            rnd_nd2=get_random_node(x_min, x_max, y_min, y_max, start_point)
            near_index1 = get_nearest_node_index(node_list1, rnd_nd1)
            near_index2 = get_nearest_node_index(node_list2, rnd_nd2)
            new_nd1 = vf_generate_new_node(node_list1[near_index1], rnd_nd1, extend_length, vector_field)
            print("new_nd1:", new_nd1.x, new_nd1.y)
            # 判断新节点1是否在地图内
            if x_min >= new_nd1.x or new_nd1.x >= x_max or y_min >= new_nd1.y or new_nd1.y >= y_max:
                continue
            new_nd2 = vf_generate_new_node(node_list2[near_index2], rnd_nd2, extend_length, vector_field)
            print("new_nd2:", new_nd2.x, new_nd2.y)
            # 判断新节点2是否在地图内
            if x_min >= new_nd2.x or new_nd2.x >= x_max or y_min >= new_nd2.y or new_nd2.y >= y_max:
                continue

            # TODO 转角约束
            if new_nd1 and not check_collision(new_nd1, node_list1[near_index1], obs_list):
                print("right_new_nd1:", new_nd1.x, new_nd1.y)
                parent_index = rewrite_index(new_nd1, node_list1, obs_list)
                new_nd1.parent = node_list1[parent_index]
                node_list1.append(new_nd1)
                new_nd1.cost = new_nd1.parent.cost + calc_p2p_dis(new_nd1, new_nd1.parent)
                rewire(new_nd1, node_list1, obs_list)
                plt.plot(new_nd1.x, new_nd1.y, "xg")
                plt.plot([new_nd1.parent.x, new_nd1.x], [new_nd1.parent.y, new_nd1.y], 'g')
            if new_nd2 and not check_collision(new_nd2, node_list2[near_index2], obs_list):
                print("right_new_nd2:", new_nd2.x, new_nd2.y)
                parent_index = rewrite_index(new_nd2, node_list2, obs_list)
                new_nd2.parent = node_list2[parent_index]
                node_list2.append(new_nd2)
                new_nd2.cost = new_nd2.parent.cost + calc_p2p_dis(new_nd2, new_nd2.parent)
                rewire(new_nd2, node_list2, obs_list)
                plt.plot(new_nd2.x, new_nd2.y, "xb")
                plt.plot([new_nd2.parent.x, new_nd2.x], [new_nd2.parent.y, new_nd2.y], 'b')

            plt.axis("equal")
            plt.axis([0.0, 260.0, -200.0, 10.0])
            for node1 in node_list1:
                if calc_p2p_dis(node1, new_nd2) <= extend_length and not check_collision(node1, new_nd2, obs_list):
                    path_found = True
                    path1 = []
                    node = node1
                    while node:
                        path1.append([node.x, node.y])
                        node = node.parent
                    path1.reverse()
                    path2 = []
                    node = new_nd2
                    while node:
                        path2.append([node.x, node.y])
                        node = node.parent
                    paths.append(path1 + path2)
                    path_num -= 1

            for node2 in node_list2:
                if calc_p2p_dis(new_nd1, node2) <= extend_length and not check_collision(new_nd1, node2, obs_list):
                    path_found = True
                    path1 = []
                    node = new_nd1
                    while node:
                        path1.append([node.x, node.y])
                        node = node.parent
                    path1.reverse()
                    path2 = []
                    node = node2
                    while node:
                        path2.append([node.x, node.y])
                        node = node.parent
                    paths.append(path1 + path2)
                    path_num -= 1
            # 找到路径后跳出循环并记录路径数量
            if path_found:
                path_num-=1
                break

    if paths:
        best_path = choose_lowest_cost(paths, vector_field)
        pruned_path = vf_prune_path(best_path, obs_list, vector_field)
        return pruned_path
    return None
