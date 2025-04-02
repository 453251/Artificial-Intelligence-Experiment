import random

import numpy as np

from utils import Individual, city_dist_mat_1, city_dist_mat_2
import matplotlib.pyplot as plt
import copy


class GeneticAlgorithm:
    def __init__(self, city_num, city_dist_mat: np.array, pop_size=50, generations=500,
                 crossover_rate=0.8, mutation_rate=0.0001):
        self.city_num = city_num
        self.city_dist_mat = city_dist_mat
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = [Individual(city_dist_mat=self.city_dist_mat, city_num=self.city_num)
                           for i in range(self.pop_size)]
        # self.best = []
        self.best = max(self.population, key=lambda individual: individual.evaluate_fitness())
        self.best_history = [self.best.calculate_distance()]

    def tournament_selection(self, input_population: list[Individual]) -> list[Individual]:
        """锦标赛选择算法"""
        population = []
        for i in range(self.pop_size):
            selected = random.sample(input_population, k=5)
            population.append(max(selected, key=lambda individual: individual.evaluate_fitness()))
        return population

    def roulette_wheel_selection(self, input_population: list[Individual]) -> list[Individual]:
        """轮盘赌选择算法"""
        total_fitness = 0
        fitness_list = []
        for individual in input_population:
            # 累积概率
            total_fitness += individual.evaluate_fitness()
            fitness_list.append(total_fitness)
        # 归一化
        probs = [prob / total_fitness for prob in fitness_list]
        population = []
        for i in range(self.pop_size):
            rand = random.uniform(0, 1)
            for k in range(len(probs)):
                if k == 0:
                    if rand < probs[k]:
                        population.append(input_population[k])
                else:
                    if probs[k - 1] <= rand < probs[k]:
                        population.append(input_population[k])
        return population

    @staticmethod
    def order_crossover(parent1_genes: list[int], parent2_genes: list[int]) -> tuple[list[int], list[int]]:
        """顺序交叉OX"""
        size = len(parent1_genes)
        # 保证起点是0
        start = random.randint(1, size - 2)
        end = random.randint(start + 1, size - 1)
        child1 = [None] * size
        child2 = [None] * size

        # 将父代中[start, end]区间的基因复制到子代中
        child1[start:end + 1] = parent1_genes[start:end + 1]
        child2[start:end + 1] = parent2_genes[start:end + 1]
        # 保证起点是0
        child1[0] = 0
        child2[0] = 0

        # 子代1：从父代2中按顺序（这里的顺序指的是从end+1开始的顺序）填充未出现的基因
        current_index = (end + 1) % size
        parent_index = (end + 1) % size
        # 保证起点是0
        while None in child1[1:]:
            if current_index == 0:
                current_index += 1
            if parent_index == 0:
                parent_index += 1
            gene = parent2_genes[parent_index]
            if gene not in child1:
                child1[current_index] = gene
                current_index = (current_index + 1) % size
            parent_index = (parent_index + 1) % size

        # 子代2：从父代1中按顺序填充未出现的基因
        current_index = (end + 1) % size
        parent_index = (end + 1) % size
        # 保证起点是0
        while None in child2[1:]:
            if current_index == 0:
                current_index += 1
            if parent_index == 0:
                parent_index += 1
            gene = parent1_genes[parent_index]
            if gene not in child2:
                child2[current_index] = gene
                current_index = (current_index + 1) % size
            parent_index = (parent_index + 1) % size

        return child1, child2

    def _crossover(self):
        """顺序交叉OX"""
        parents_0 = self.population[0:self.pop_size // 2]
        parents_1 = self.population[self.pop_size // 2:]
        intersection = set(parents_0) & set(parents_1)
        parents_0 = [p for p in parents_0 if p not in intersection]
        parents_1 = [p for p in parents_1 if p not in intersection]
        select_number = int(self.pop_size * self.crossover_rate)
        parents_0 = random.sample(parents_0, k=min(len(parents_0), select_number // 2))
        parents_1 = random.sample(parents_1, k=min(len(parents_1), select_number // 2))
        remainder = [ind for ind in self.population if ind not in parents_0 and ind not in parents_1]

        offspring = []
        for parent1, parent2 in zip(parents_0, parents_1):
            child1_genes, child2_genes = self.order_crossover(parent1.genes, parent2.genes)
            offspring.append(Individual(self.city_dist_mat, child1_genes, self.city_num))
            offspring.append(Individual(self.city_dist_mat, child2_genes, self.city_num))
            # child1 = Individual(self.city_dist_mat, child1_genes, self.city_num)
            # child2 = Individual(self.city_dist_mat, child2_genes, self.city_num)
            # if child1.evaluate_fitness() > parent1.evaluate_fitness():
            #     offspring.append(child1)
            # else:
            #     offspring.append(parent1)
            # if child2.evaluate_fitness() > parent2.evaluate_fitness():
            #     offspring.append(child2)
            # else:
            #     offspring.append(parent2)

        return remainder + offspring
    #
    # def _crossover(self) -> list[Individual]:
    #     num_to_crossover = int(self.crossover_rate * self.pop_size)
    #     # 保证选出的个体数为偶数
    #     if num_to_crossover % 2 == 1:
    #         num_to_crossover -= 1
    #     # # 剩下不去做交叉的，选取num_remainder个最优的，保证最优
    #     # num_remainder = self.pop_size - num_to_crossover
    #     # population = copy.deepcopy(self.population)
    #     # population.sort(key=lambda ind: ind.evaluate_fitness(), reverse=False)
    #     # remainder = population[:num_remainder]
    #     num_remainder = self.pop_size - num_to_crossover
    #     remainder = random.sample(self.population, num_remainder)
    #     offspring = []
    #     for _ in range(num_to_crossover // 2):
    #         parent1 = random.choice(self.population)
    #         parent2 = random.choice(self.population)
    #         attempts = 0
    #         # 如果两个父代完全相同，尝试重新抽取最多5次
    #         while parent1.genes == parent2.genes and attempts < 5:
    #             parent2 = random.choice(self.population)
    #             attempts += 1
    #         # 若多次尝试仍得到相同个体，则对parent2进行轻微变异
    #         if parent1.genes == parent2.genes:
    #             parent2_genes = parent1.genes[:]  # 克隆父代1的基因
    #             # 采用交换变异方式
    #             idx1, idx2 = random.sample(range(1, self.city_num), 2)
    #             parent2_genes[idx1], parent2_genes[idx2] = parent2_genes[idx2], parent2_genes[idx1]
    #             parent2 = Individual(self.city_dist_mat, genes=parent2_genes, city_num=self.city_num)
    #         child1_genes, child2_genes = self.order_crossover(parent1.genes, parent2.genes)
    #         # offspring.append(Individual(self.city_dist_mat, genes=child1_genes, city_num=self.city_num))
    #         # offspring.append(Individual(self.city_dist_mat, genes=child2_genes, city_num=self.city_num))
    #         child1 = Individual(self.city_dist_mat, child1_genes, self.city_num)
    #         child2 = Individual(self.city_dist_mat, child2_genes, self.city_num)
    #         offspring.append(child1)
    #         offspring.append(child2)
    #         # if child1.evaluate_fitness() > parent1.evaluate_fitness():
    #         #     offspring.append(child1)
    #         # else:
    #         #     offspring.append(parent1)
    #         # if child2.evaluate_fitness() > parent2.evaluate_fitness():
    #         #     offspring.append(child2)
    #         # else:
    #         #     offspring.append(parent2)
    #     return remainder + offspring

    def _mutation(self, population: list[Individual]) -> list[Individual]:
        """
            交换变异，这里采用了精英保留策略，由于本身保留了父代的最优解
            且保证变异后的子代优于父代，使得遗传算法能够严格收敛
            并且由于变异后的子代优于父代，所以变异率也可以设置很大，从而加强跳出局部最优解的可能
        """
        mutation_number = int(self.pop_size * self.mutation_rate)
        mutation_candidates = random.sample(population, k=mutation_number)
        remainder = [ind for ind in population if ind not in mutation_candidates]
        # # 剩下不去做交叉的，选取num_remainder个最优的，保证最优
        # num_remainder = self.pop_size - mutation_number
        # population_copy = copy.deepcopy(population)
        # population_copy.sort(key=lambda ind: ind.evaluate_fitness(), reverse=False)
        # remainder = population_copy[:num_remainder]
        mutation = []
        for individual in mutation_candidates:
            idx1, idx2 = random.sample(range(1, self.city_num), 2)
            individual_copy = copy.deepcopy(individual)
            individual_copy.genes[idx1], individual_copy.genes[idx2] = \
                individual_copy.genes[idx2], individual_copy.genes[idx1]
            mutation.append(individual_copy)
            # if individual_copy.evaluate_fitness() > individual.evaluate_fitness():
            #     mutation.append(individual_copy)
            # else:
            #     mutation.append(individual)

        return remainder + mutation

    def next_generation(self):
        pool = self._crossover()
        pool = self._mutation(pool)
        # self.population = self.tournament_selection(pool)
        self.population = self.roulette_wheel_selection(pool)
        # print(len(self.population))
        # 更新全局最佳个体
        current_best = max(self.population, key=lambda individual: individual.evaluate_fitness())
        # self.best.append([current_best, current_best.calculate_distance()])
        if current_best.evaluate_fitness() > self.best.evaluate_fitness():
            self.best = current_best
        self.best_history.append(self.best.calculate_distance())

    def train(self):
        for generation in range(self.generations):
            self.next_generation()
            # print(f"Generation {generation + 1}: Best distance = {self.best[-1][1]}")
            print(f"Generation {generation + 1}: Best distance = {self.best.calculate_distance()}")
        return self.best, self.best_history

    def plot_convergence(self):
        """绘制收敛过程"""
        plt.figure(figsize=(10, 5))
        plt.plot(range(0, self.generations + 1), self.best_history, marker='o', linestyle='-', markersize=4)
        plt.xlabel("Generation")
        plt.ylabel("Best Distance")
        plt.title(f"Genetic Algorithm Convergence for TSP(crossover_ratio:{self.crossover_rate}, "
                  f"mutation_ratio:{self.mutation_rate})")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    city_num_1 = 5
    # 实验一：5城市TSP问题
    print("====== Experiment 1: 5-city TSP ======")
    ga1 = GeneticAlgorithm(city_dist_mat=city_dist_mat_1, city_num=city_num_1)
    best_ind1, history1 = ga1.train()
    print("Best route found:")
    print(best_ind1)
    ga1.plot_convergence()

    # best = ga1.train()
    # best_individual, min_distance = min(best, key=lambda f: f[1])  # 直接获取最短路径
    # best.sort(key=lambda f: f[1], reverse=True)
    # print("Best route found:")
    # print(best_individual)
    # ga1.plot_convergence(best)
    print("\n-----------------------------\n")

    # 实验二：8城市TSP问题
    print("====== Experiment 2: 8-city TSP ======")
    ga2 = GeneticAlgorithm(city_dist_mat=city_dist_mat_2, city_num=8)
    best_ind2, history2 = ga2.train()
    print("Best route found:")
    print(best_ind2)
    ga2.plot_convergence()
    # best = ga2.train()
    # best_individual, min_distance = min(best, key=lambda f: f[1])  # 直接获取最短路径
    # best.sort(key=lambda f: f[1], reverse=True)
    # print("Best route found:")
    # print(best_individual)
    # ga1.plot_convergence(best)
