# 类似Bi_RRT & Bi_RRT*子类，考虑流场情况下的rrt，新增生成新点、计算上游系数并选择最小cost路径的函数
import numpy as np
from numpy.lib.utils import source

from single_planning.Bi_RRT_star import *
from utils.node import Node


def get_changed_direction(q_near, q_rand, F, lambda_param=0.1):
    """论文中的GetChangedDirection逻辑"""
    # 计算随机方向向量v_rand
    dx_rand = q_rand.x - q_near.x
    dy_rand = q_rand.y - q_near.y
    v_rand = np.array([dx_rand, dy_rand])
    norm_rand = np.linalg.norm(v_rand)
    if norm_rand < 1e-6:
        return np.zeros(2)
    v_rand = v_rand / norm_rand

    # 计算向量场方向v_field
    F_norm = np.linalg.norm(F)
    if F_norm < 1e-6:
        return v_rand
    v_field = F / F_norm

    # 计算夹角θ_rand
    cos_theta = np.dot(v_rand, v_field)
    theta_rand = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # 计算上游成本U_rand
    U_rand = F_norm * (1 - cos_theta)

    # 指数分布逆变换计算U_new
    if F_norm < 1e-6 or lambda_param < 1e-6:
        U_new = U_rand
    else:
        numerator = U_rand * (1 - np.exp(-2 * lambda_param * F_norm))
        denominator = 2 * F_norm
        U_new = -np.log(1 - numerator / denominator) / lambda_param if denominator > 1e-6 else 0

    # 计算θ_new和权重w
    U_new_clamped = min(U_new, 2 * F_norm)
    cos_theta_new = 1 - U_new_clamped / F_norm
    theta_new = np.arccos(np.clip(cos_theta_new, -1.0, 1.0))

    if np.sin(theta_new) < 1e-6:
        w = 0.0
    else:
        w = np.sin(theta_rand - theta_new) / np.sin(theta_new)

    # 合成新方向
    v_new = v_rand + w * v_field
    return v_new / np.linalg.norm(v_new) if np.linalg.norm(v_new) > 1e-6 else v_rand


# 结合向量场方向得到新节点
# 随机方向归一化，向量保持原来的大小，两者加权平均
def vf_generate_new_node(nearest_node, random_node, extend_length, vector_field, is_start_tree,lambda_param):
    # # 获取最近节点到随机点的方向向量
    # direction_to_random = np.array([random_node.x - nearest_node.x, random_node.y - nearest_node.y])
    # # 不该出现0的情况（原来是回旋镖）
    # if np.linalg.norm(direction_to_random) != 0:
    #     direction_to_random /= np.linalg.norm(direction_to_random)
    # # print("nearest_node:", nearest_node.x, nearest_node.y)
    #
    # # 获取最近节点处的向量场方向
    # u, v = get_vector_field(nearest_node.x, nearest_node.y, vector_field)
    # direction_vector_field = np.array([u, v])
    #
    # # 终点树向量场反向
    # if not is_start_tree:
    #     direction_vector_field = -direction_vector_field
    #
    # # 两个方向向量的加权平均方向
    # average_direction_vector = (direction_to_random + direction_vector_field) / 2
    #
    # # 归一化平均方向向量
    # # average_direction_vector /= np.linalg.norm(average_direction_vector)
    #
    # # 根据这个平均方向生成新节点
    # new_x = nearest_node.x + extend_length * average_direction_vector[0]
    # new_y = nearest_node.y + extend_length * average_direction_vector[1]

    # 获取向量场（需保持原有坐标系处理逻辑）
    u, v = get_vector_field(nearest_node.x, nearest_node.y, vector_field)
    F = np.array([u, v])

    # 终点树向量场反向
    if not is_start_tree:
        F = -F

    # 调用论文方法生成方向
    v_new = get_changed_direction(
        q_near=nearest_node,
        q_rand=random_node,
        F=F,
        lambda_param=lambda_param  # 可调整为参数
    )

    # 生成新节点
    new_x = nearest_node.x + v_new[0] * extend_length
    new_y = nearest_node.y + v_new[1] * extend_length

    new_node = Node(new_x, new_y)
    new_node.parent = nearest_node
    return new_node


# 得到该位置向量场方向——插值
# 向量大小
def get_vector_field(x, y, vector_field):
    X, Y, U, V = vector_field
    x1 = int(np.floor(x)) + 20
    # 向量场的坐标系是从-220到20,U V的坐标系是从0开始
    y1 = int(np.floor(y)) + 220
    # x2 = x1 + 1
    # y2 = y1 + 1
    # 这里不用改
    # u = U[y1, x1]*(x2-x) * (y2-y) + U[y1, x2]*(x-x1) * (y2-y) + U[y2, x1]*(x2-x) * (y-y1) + U[y2, x2]*(x-x1) * (y-y1)
    # v = V[y1, x1]*(x2-x) * (y2-y) + V[y1, x2]*(x-x1) * (y2-y) + V[y2, x1]*(x2-x) * (y-y1) + V[y2, x2]*(x-x1) * (y-y1)
    u = U[y1, x1]
    v = V[y1, x1]
    return u, v


# 计算路径中每一步的上游系数结果（rrt-dwa方法）
# 向量大小
def upstream_criterion(path, vector_field):
    total_difference = 0
    for i in range(1, len(path)):
        start_point = path[i - 1]
        end_point = path[i]
        distance = np.linalg.norm(np.array(end_point) - np.array(start_point))
        num_points = int(distance / 1.4) + 1  # 每1.4米插入一个点==>向量场为 1m×1m，对角线即1.4米
        x_values = np.linspace(start_point[0], end_point[0], num_points)
        y_values = np.linspace(start_point[1], end_point[1], num_points)
        for x, y in zip(x_values, y_values):
            # 当前位置的向量场
            u, v = get_vector_field(x, y, vector_field)
            vector_field_magnitude = np.sqrt(u ** 2 + v ** 2)
            # vector_field_magnitude = 1
            # 不做归一化，保留向量场大小
            direction_vector_field = np.array([u, v]) / 1

            # 当前速度方向——路径求导
            direction_path = np.gradient(np.array(path), axis=0)[i]
            path_magnitude = np.linalg.norm(direction_path)
            # 归一化速度方向 --> 需要，因为向量场不做归一化 --> 应用不等式时两者模长相等，均为1
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
        # print("cost:", cost)
        if cost < min_cost and cost is not None:
            min_cost = cost
            best_path = path
    return best_path


# 考虑流场的重写-->不做了，这个运行太浪费时间
def vf_rewrite_index(new_node, node_list, obs_list, path_node, vector_field):
    # 重写的条件：1.在地图内 2.不碰撞 3.最小cost
    r = 8
    min_score = float('inf')
    min_node_index = None
    for i, node in enumerate(node_list):
        if calc_p2p_dis(new_node, node) <= r and not check_collision(new_node, node, obs_list):
            if node.parent is not None:
                degree = calc_triangle_deg(node, new_node, node.parent)
                if degree < 90:
                    continue
            potential_score = path_score(path_node + [new_node, node], vector_field)
            if potential_score < min_score:
                min_score = potential_score
                min_node_index = i
    return min_node_index


# 考虑流场的重布线
def vf_rewire(node_new, node_list, obstacle_list, vector_field):
    r = 15
    for node in node_list:
        if (node != node_new.parent and calc_p2p_dis(node_new, node) < r
                and not check_collision(node_new, node, obstacle_list)):
            if node_new.parent is not None:
                degree = calc_triangle_deg(node_new, node, node_new.parent)
                if degree <= 90:
                    continue
            potential_cost = path_score([node_new, node], vector_field)
            if potential_cost < node.cost:
                node.parent = node_new
                node.cost = potential_cost


# 考虑流场的剪枝
def vf_prune_path(path, obs_list, vector_field):
    mini_degree = 90  # 最大转角，180°-mini_degree，反着定义
    pruned_path = [path[0]]
    i = 0

    while i < len(path) - 1:
        found = False
        for j in range(len(path) - 1, i, -1):
            # 转角约束
            if i > 0:
                # 这里转角之前定义错了
                degree = calc_triangle_deg(path[i], pruned_path[-2], path[j])
                if degree < mini_degree and degree != 0:
                    continue
            # 碰撞约束
            if not check_collision(path[i], path[j], obs_list):
                # 保证至少有一个点满足转角约束
                if j + 1 < len(path):
                    degree_bot = calc_triangle_deg(path[j], path[i], path[j + 1])
                    if degree_bot < mini_degree and degree_bot != 0:
                        continue
                # 现有优化路径 + 现在考虑的路径不剪枝形式
                candidate_path = pruned_path + path[i + 1:]
                # 计算路径评分——跳过 i 到 j 中间的路径
                score_jump = path_score(pruned_path + path[j:], vector_field)
                # 计算路径评分——不跳
                score_old = path_score(candidate_path, vector_field)
                # print("score_jump:", score_jump, " score_old:", score_old)
                # print("pruned_path:", pruned_path, " i:", i)
                # Compare the scores,分数越小越好
                if score_jump - score_old <= 0 and score_jump != 0:
                    pruned_path.append(path[j])
                    i = j
                    found = True
                    break
        if not found:
            if i + 1 < len(path):
                pruned_path.append(path[i + 1])
            i += 1
    if pruned_path[-1] != path[-1]:
        pruned_path.append(path[-1])
    return pruned_path


# 剪枝中的评分函数
# 向量大小
def path_score(path, vector_field):
    # print("score path:", path)
    # 累计diff 向量场方向与路径方向的差异
    # 算完之后乘这个位置向量的大小，之后叠加得sum
    total_difference = 0
    # 累计转角
    total_angle = 0
    # 累计长度
    total_length = 0
    direction_path = 0
    # 评分细则（+转角考虑）
    for i in range(len(path) - 1):  # 最大值是 len(path) - 2
        start_point = path[i]
        end_point = path[i + 1]

        # 计算路径段的距离
        distance = np.linalg.norm(np.array(end_point) - np.array(start_point))
        total_length += distance
        # 根据距离确定点的数量，向量场为 1m x 1m 的网格
        num_points = int(distance) + 1
        # 路径段上的每个点作区划，得到累计diff
        x_values = np.linspace(start_point[0], end_point[0], num_points)
        y_values = np.linspace(start_point[1], end_point[1], num_points)
        x_values = x_values[:num_points - 1]
        y_values = y_values[:num_points - 1]
        direction_path = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        for x, y in zip(x_values, y_values):
            u, v = get_vector_field(x, y, vector_field)
            if np.isnan(u) or np.isnan(v):
                continue
            direction_vector_field = np.arctan2(v, u)
            total_difference += abs(direction_path - direction_vector_field) * np.linalg.norm([u, v])
            # if np.isnan(total_difference):
            #     print("u:", u, " v:", v, " direction_vector_field:", direction_vector_field, " direction_path:",
            #           direction_path, " start:", start_point, " end:", end_point)
        # print("total_difference:", total_difference)
        if i < len(path) - 2:
            angle = abs(np.arctan2(path[i + 2][1] - path[i + 1][1], path[i + 2][0] - path[i + 1][0]) - direction_path)
            total_angle += angle
        # print("total_angle:", total_angle)
        # 归一化并赋权重
        # total_length /= 350
        # total_difference /= 150
        # total_angle /= 20
    prize_length = 0.2
    prize_difference = 1
    prize_angle = 0.2

    # print("total_length:", total_length, " total_difference:", total_difference, " total_angle:", total_angle)
    return total_difference * prize_difference + total_angle * prize_angle + total_length * prize_length


def VF_Bi_RRT_star_plan(start_xy, goal_xy, obslis_xy, vector_field,lambda_param=3):
    """设置lamda_param越小，越不考虑向量场的影响，采用了VF-RRT原生方法"""
    x_min = min(start_xy[0], goal_xy[0]) - 10
    x_max = max(start_xy[0], goal_xy[0]) + 10
    y_min = min(start_xy[1], goal_xy[1]) - 10
    y_max = max(start_xy[1], goal_xy[1]) + 10
    start_point = start_xy
    goal_point = goal_xy
    obs_list = obslis_xy
    extend_length = 5  # 要走迷宫的话改为10，rewrite里的半径改为15
    max_iter = 10000
    mini_degree = 90  # 最大转角，180°-mini_degree，反着定义
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
        path_found = False
        for i in range(max_iter):
            rnd_nd1 = get_random_node(x_min, x_max, y_min, y_max, goal_point)
            rnd_nd2 = get_random_node(x_min, x_max, y_min, y_max, start_point)
            near_index1 = get_nearest_node_index(node_list1, rnd_nd1)
            near_index2 = get_nearest_node_index(node_list2, rnd_nd2)
            new_nd1 = vf_generate_new_node(node_list1[near_index1], rnd_nd1, extend_length, vector_field,
                                           is_start_tree=True,lambda_param=lambda_param)
            # print("new_nd1:", new_nd1.x, new_nd1.y)
            # 判断新节点1是否在地图内
            if (x_min >= new_nd1.x or new_nd1.x >= x_max or y_min >= new_nd1.y or new_nd1.y >= y_max
                    or new_nd1.x is None):
                continue
            new_nd2 = vf_generate_new_node(node_list2[near_index2], rnd_nd2, extend_length, vector_field,
                                           is_start_tree=False,lambda_param=lambda_param)
            # print("new_nd2:", new_nd2.x, new_nd2.y)
            # 判断新节点2是否在地图内
            if (x_min >= new_nd2.x or new_nd2.x >= x_max or y_min >= new_nd2.y or new_nd2.y >= y_max
                    or new_nd2.x is None):
                continue

            # 转角约束
            if node_list1[near_index1].parent is not None and node_list2[near_index2].parent is not None:
                degree1 = calc_triangle_deg(node_list1[near_index1], rnd_nd1, node_list1[near_index1].parent)
                degree2 = calc_triangle_deg(node_list2[near_index2], rnd_nd2, node_list2[near_index2].parent)
                if degree1 < mini_degree and degree1 != 0 or degree2 < mini_degree and degree2 != 0:
                    continue
            # else:
            #     continue

            if new_nd1 and not check_collision(new_nd1, node_list1[near_index1], obs_list):
                # print("right_new_nd1:", new_nd1.x, new_nd1.y)
                parent_index = rewrite_index(new_nd1, node_list1, obs_list)
                if parent_index is None:
                    parent_index = near_index1
                new_nd1.parent = node_list1[parent_index]
                node_list1.append(new_nd1)
                new_nd1.cost = new_nd1.parent.cost + calc_p2p_dis(new_nd1, new_nd1.parent)
                rewire(new_nd1, node_list1, obs_list)
                plt.plot(new_nd1.x, new_nd1.y, "xg")
                plt.plot([new_nd1.parent.x, new_nd1.x], [new_nd1.parent.y, new_nd1.y], 'g')
            else:
                continue
            if new_nd2 and not check_collision(new_nd2, node_list2[near_index2], obs_list):
                # print("right_new_nd2:", new_nd2.x, new_nd2.y)
                parent_index = rewrite_index(new_nd2, node_list2, obs_list)
                if parent_index is None:
                    parent_index = near_index2
                new_nd2.parent = node_list2[parent_index]
                node_list2.append(new_nd2)
                new_nd2.cost = new_nd2.parent.cost + calc_p2p_dis(new_nd2, new_nd2.parent)
                rewire(new_nd2, node_list2, obs_list)
                plt.plot(new_nd2.x, new_nd2.y, "xb")
                plt.plot([new_nd2.parent.x, new_nd2.x], [new_nd2.parent.y, new_nd2.y], 'b')
            else:
                continue

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
                    path = path1 + path2
                    # print("上个路径：",path)
                    paths.append(path)
                    # path_num -= 1
                    break

            for node2 in node_list2:
                if calc_p2p_dis(new_nd1, node2) <= extend_length and not check_collision(new_nd1, node2, obs_list):
                    path_found = True
                    path2 = []
                    node = node2
                    while node:
                        path2.append([node.x, node.y])
                        node = node.parent
                    # path2.reverse()
                    path1 = []
                    node = new_nd1
                    while node:
                        path1.append([node.x, node.y])
                        node = node.parent
                    path1.reverse()
                    path = path1 + path2
                    # path.reverse()
                    paths.append(path)
                    # print("这个路径 ",path)
                    # path_num -= 1
                    break

            # 找到路径后跳出循环并记录路径数量
            if path_found:
                path_num -= 1
                break

    if paths:
        best_path = choose_lowest_cost(paths, vector_field)
        # print("best_path:", best_path)
        pruned_path = vf_prune_path(best_path, obs_list, vector_field)
        return pruned_path
        # return best_path
    return None
