import pygame
import heapq
import numpy as np

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 20, 20
TILE_SIZE = WIDTH // COLS
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Pathfinding Visualization Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)  # For visualizing the open list
ORANGE = (255, 165, 0)   # For visualizing the closed list

# Node class to represent each cell in the grid
class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0  # Cost from start node
        self.h = 0  # Heuristic cost (Manhattan distance)
        self.f = 0  # Total cost (g + h)

    def __lt__(self, other):
        return self.f < other.f

def heuristic(node, goal):
    return abs(node.x - goal.x) + abs(node.y - goal.y)

def a_star(grid, start, goal):
    open_list = []
    closed_list = set()
    open_dict = {}  # For visualizing nodes in open list
    closed_dict = {}  # For visualizing nodes in closed list

    heapq.heappush(open_list, (start.f, start))
    open_dict[(start.x, start.y)] = True  # Mark node in open list
    
    while open_list:
        _, current_node = heapq.heappop(open_list)
        if current_node.x == goal.x and current_node.y == goal.y:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1], open_dict, closed_dict  # Return reversed path and visualization data

        closed_list.add((current_node.x, current_node.y))
        closed_dict[(current_node.x, current_node.y)] = True  # Mark node in closed list

        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for n in neighbors:
            neighbor_x, neighbor_y = current_node.x + n[0], current_node.y + n[1]
            if 0 <= neighbor_x < grid.shape[0] and 0 <= neighbor_y < grid.shape[1] and grid[neighbor_x][neighbor_y] == 0:
                neighbor_node = Node(neighbor_x, neighbor_y, current_node)
                if (neighbor_x, neighbor_y) in closed_list:
                    continue
                neighbor_node.g = current_node.g + 1
                neighbor_node.h = heuristic(neighbor_node, goal)
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                if (neighbor_x, neighbor_y) not in open_dict:
                    heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
                    open_dict[(neighbor_x, neighbor_y)] = True  # Mark node in open list

        # Adding delay of 0.5 seconds between iterations
        pygame.time.delay(100)  
        visualize_search(open_dict, closed_dict)  # Visualize open and closed nodes
        draw_grid()
        draw_obstacles(grid)
        pygame.display.flip()  # Update display after each iteration

    return None, open_dict, closed_dict  # No path found

def draw_grid():
    for row in range(ROWS):
        for col in range(COLS):
            pygame.draw.rect(screen, GRAY, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 1)

def draw_obstacles(grid):
    for row in range(ROWS):
        for col in range(COLS):
            if grid[row][col] == 1:
                pygame.draw.rect(screen, BLACK, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE))

def draw_path(path):
    if path:
        for (x, y) in path:
            pygame.draw.rect(screen, GREEN, (y * TILE_SIZE, x * TILE_SIZE, TILE_SIZE, TILE_SIZE))

def create_obstacles(grid):
    # Adding some fixed obstacles (shape of L and squares)
    grid[5:8, 5] = 1  # Vertical part of L
    grid[7, 6:8] = 1  # Horizontal part of L
    grid[10:13, 10:12] = 1  # Square obstacle
    return grid

def visualize_search(open_dict, closed_dict):
    for (x, y) in open_dict:
        pygame.draw.rect(screen, YELLOW, (y * TILE_SIZE, x * TILE_SIZE, TILE_SIZE, TILE_SIZE))
    for (x, y) in closed_dict:
        pygame.draw.rect(screen, ORANGE, (y * TILE_SIZE, x * TILE_SIZE, TILE_SIZE, TILE_SIZE))

def main():
    run = True
    clock = pygame.time.Clock()

    # Create grid and obstacles
    grid = np.zeros((ROWS, COLS), dtype=int)
    grid = create_obstacles(grid)

    # Player and goal positions
    start = Node(0, 0)
    goal = Node(ROWS - 1, COLS - 1)

    # Find path using A* and get visualization data
    path, open_dict, closed_dict = a_star(grid, start, goal)
    current_step = 0
    path_found = False

    while run:
        screen.fill(WHITE)
        draw_grid()
        draw_obstacles(grid)

        # Visualize search process (yellow = open list, orange = closed list)
        if not path_found:
            visualize_search(open_dict, closed_dict)

        # When path is found, draw the path and move the character
        if path and path_found:
            draw_path(path[:current_step])
            pygame.draw.rect(screen, RED, (goal.y * TILE_SIZE, goal.x * TILE_SIZE, TILE_SIZE, TILE_SIZE))  # Goal
            pygame.draw.rect(screen, BLUE, (path[current_step][1] * TILE_SIZE, path[current_step][0] * TILE_SIZE, TILE_SIZE, TILE_SIZE))  # Main character

            if current_step < len(path) - 1:
                current_step += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        pygame.display.flip()

        # Wait for a moment after visualizing the search, then start moving the character
        if not path_found:
            pygame.time.wait(2000)  # Wait for 2 seconds after visualizing search
            path_found = True  # Start the movement
        else:
            clock.tick(5)  # Speed of movement after path found

    pygame.quit()

if __name__ == "__main__":
    main()
