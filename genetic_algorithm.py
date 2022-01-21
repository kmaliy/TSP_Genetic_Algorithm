import pygame
import matplotlib.pyplot as plt

import time
from constants import BLACK
from helpers import make_grid, initial_population, rank_routes, next_generation, draw, get_clicked_pos, City


ROWS = 60
WIDTH = 600
GRID = make_grid(ROWS, WIDTH)

WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption('TSP Genetic Algorithm')


def genetic_algorithm(population, population_size, elite_size, mutation_rate, generations):
    pop = initial_population(population_size, population)
    print("Initial distance: " + str(1 / rank_routes(pop)[0][1]))
    progress = list()
    start_time = time.time()
    progress.append(1 / rank_routes(pop)[0][1])

    for i in range(0, generations):
        pop = next_generation(pop, elite_size, mutation_rate)
        progress.append(1 / rank_routes(pop)[0][1])
        route = pop[rank_routes(pop)[0][0]]
        end_time = time.time()
        print(str(end_time-start_time) + " Generation: " + str(i) + " Distance: " + str(progress[i]) + " Route: " + str(route))
        route_points = [(city.x, city.y) for city in route]
        draw(WIN, GRID, route_points)

    print("Final distance: " + str(1 / rank_routes(pop)[0][1]))
    best_route_index = rank_routes(pop)[0][0]
    best_route = pop[best_route_index]
    print(best_route)
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

    return best_route


def main():
    list_of_spots = []
    best_route = []

    start = None
    end = None

    run = True
    while run:
        draw(WIN, GRID, best_route)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:  # press the left button of the mouse
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, WIDTH)
                spot = GRID[row][col]
                if not start and spot != end:
                    start = spot
                    start.make_start_city()

                elif spot != start:
                    spot.make_city()

                list_of_spots = [
                    (int(spot.x), int(spot.y))
                    for row in GRID for spot in row
                    if spot.color == BLACK
                ]

            elif pygame.mouse.get_pressed()[2]:  # press the right button of the mouse
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, WIDTH)
                spot = GRID[row][col]
                spot.reset()
                if spot == start:
                    start = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start:
                    city_list = [City(x=spot[0], y=spot[1]) for spot in list_of_spots]

                    print(city_list)
                    best_route = genetic_algorithm(population=city_list, population_size=100, elite_size=30,
                                                   mutation_rate=0.01, generations=1000)
                    best_route = [(spot.x, spot.y) for spot in best_route]

    pygame.quit()


if __name__ == '__main__':
    main()
