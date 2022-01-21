import operator
import pandas as pd
import numpy as np
import pygame
import random

from constants import WHITE, BLACK, PURPLE


class Spot:
    def __init__(self, row, col, width, total_spots):
        self.row = row
        self.col = col
        self.color = WHITE
        self.width = width
        self.x = row * width
        self.y = col * width
        self.total_spots = total_spots

    def get_position(self):
        return self.row, self.col

    def is_city(self):
        return self.color == BLACK

    def is_start_city(self):
        return self.color == BLACK

    def reset(self):
        self.color = WHITE

    def make_start_city(self):
        self.color = BLACK

    def make_city(self):
        self.color = BLACK

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def __lt__(self, other):
        return False


def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)

    return grid


def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, WHITE, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, WHITE, (j * gap, 0), (j * gap, width))


def draw(win, grid, spots):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    for index in range(0, len(spots)):
        next_index = index + 1 if index + 1 != len(spots) else 0
        pygame.draw.line(win, PURPLE, spots[index], spots[next_index], 2)

    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def route_distance(self):
        if self.distance == 0:
            path_distance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                path_distance += fromCity.distance(toCity)
            self.distance = path_distance
        return self.distance

    def route_fitness_score(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.route_distance())
        return self.fitness


def create_route(city_list):
    route = random.sample(city_list, len(city_list))
    return route


def initial_population(population_size, city_list):
    population = []

    for i in range(0, population_size):
        population.append(create_route(city_list))
    return population


def rank_routes(population):
    fitness_results = {}
    for i in range(0, len(population)):
        fitness_results[i] = Fitness(population[i]).route_fitness_score()
    return sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=True)


def selection(pop_ranked, elite_size):
    selection_results = []
    df = pd.DataFrame(np.array(pop_ranked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, elite_size):
        selection_results.append(pop_ranked[i][0])

    for i in range(0, len(pop_ranked) - elite_size):
        pick = 100 * random.random()
        for j in range(0, len(pop_ranked)):
            if pick <= df.iat[j, 3]:
                selection_results.append(pop_ranked[j][0])
                break

    return selection_results


def matingpool(population, selection_results):
    mating_pool = []
    for i in range(0, len(selection_results)):
        index = selection_results[i]
        mating_pool.append(population[index])
    return mating_pool


def breed(parent1, parent2):
    child_parent1 = []

    gene_A = int(random.random() * len(parent1))
    gene_B = int(random.random() * len(parent1))

    start_gene = min(gene_A, gene_B)
    end_gene = max(gene_A, gene_B)

    for i in range(start_gene, end_gene):
        child_parent1.append(parent1[i])

    child_parent2 = [item for item in parent2 if item not in child_parent1]

    child = child_parent1 + child_parent2
    return child


def breed_population(mating_pool, elite_size):
    children = []
    length = len(mating_pool) - elite_size
    pool = random.sample(mating_pool, len(mating_pool))

    for i in range(0, elite_size):
        children.append(mating_pool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(mating_pool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swap_with]

            individual[swapped] = city2
            individual[swap_with] = city1
    return individual


def mutate_population(population, mutation_rate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutated_individual = mutate(population[ind], mutation_rate)
        mutatedPop.append(mutated_individual)
    return mutatedPop


def next_generation(current_generation, elite_size, mutation_rate):
    pop_ranked = rank_routes(current_generation)
    selection_results = selection(pop_ranked, elite_size)
    mating_pool = matingpool(current_generation, selection_results)
    children = breed_population(mating_pool, elite_size)
    return mutate_population(children, mutation_rate)
