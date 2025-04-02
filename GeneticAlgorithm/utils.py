import numpy as np
import random

city_num_1 = 5
city_dist_mat_1 = np.zeros([city_num_1, city_num_1])
city_dist_mat_1[0][1] = city_dist_mat_1[1][0] = 1165
city_dist_mat_1[0][2] = city_dist_mat_1[2][0] = 1462
city_dist_mat_1[0][3] = city_dist_mat_1[3][0] = 3179
city_dist_mat_1[0][4] = city_dist_mat_1[4][0] = 1967
city_dist_mat_1[1][2] = city_dist_mat_1[2][1] = 1511
city_dist_mat_1[1][3] = city_dist_mat_1[3][1] = 1942
city_dist_mat_1[1][4] = city_dist_mat_1[4][1] = 2129
city_dist_mat_1[2][3] = city_dist_mat_1[3][2] = 2677
city_dist_mat_1[2][4] = city_dist_mat_1[4][2] = 1181
city_dist_mat_1[3][4] = city_dist_mat_1[4][3] = 2216

city_dist_mat_2 = np.array([
    [0, 49, 25, 19, 63, 74, 26, 39],
    [49, 0, 26, 48, 65, 36, 42, 55],
    [25, 26, 0, 26, 21, 24, 78, 49],
    [19, 48, 26, 0, 45, 44, 57, 62],
    [63, 65, 21, 45, 0, 47, 48, 54],
    [74, 36, 24, 44, 47, 0, 47, 65],
    [26, 42, 78, 57, 48, 47, 0, 47],
    [39, 55, 49, 62, 54, 65, 47, 0]])


class Individual:
    def __init__(self, city_dist_mat: np.array, genes=None, city_num=5):
        self.city_num = city_num
        self.city_dist_mat = city_dist_mat
        if genes is None:
            genes = list(range(1, city_num))  # 城市索引从 1 开始，起点（0）不变
            random.shuffle(genes)
            self.genes = [0] + genes  # 确保起点为 0
        else:
            self.genes = genes

    def calculate_distance(self):
        distance = 0
        for i in range(self.city_num - 1):
            distance += self.city_dist_mat[self.genes[i]][self.genes[i+1]]
        distance += self.city_dist_mat[self.genes[-1]][self.genes[0]]
        return distance

    def evaluate_fitness(self):
        return 1 / self.calculate_distance()

    def __str__(self):
        route = " -> ".join(str(city) for city in self.genes) + " -> 0"
        return f"Route: {route} | Distance: {self.calculate_distance()}"
