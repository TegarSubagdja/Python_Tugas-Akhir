import pygame
import heapq
import numpy as np

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 20, 20
TILE_SIZE = WIDTH // COLS
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Jump Point Search Visualization")

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

def jump(grid, node, parent, goal):
    x, y = node.x, node.y
    dx, dy = x - parent.x, y - parent.y

    # Check if out of bounds or is an obstacle
    if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]) or grid[x][y] == 1:
        return None

    # Goal found
    if (x, y) == (goal.x, goal.y):
        return node

    # Horizontal/Vertical jumps
    if dx != 0 and dy == 0:
        if (0 <= x + dx < grid.shape[0] and 0 <= y - 1 < grid.shape[1] and grid[x + dx][y - 1] == 0) or \
           (0 <= x + dx < grid.shape[0] and 0 <= y + 1 < grid.shape[1] and grid[x + dx][y + 1] == 0):
            return node
    elif dy != 0 and dx == 0:
        if (0 <= x - 1 < grid.shape[0] and 0 <= y + dy < grid.shape[1] and grid[x - 1][y + dy] == 0) or \
           (0 <= x + 1 < grid.shape[0] and 0 <= y + dy < grid.shape[1] and grid[x + 1][y + dy] == 0):
            return node

    # Diagonal jumps
    if dx != 0 and dy != 0:
        if (0 <= x - dx < grid.shape[0] and 0 <= y + dy < grid.shape[1] and grid[x - dx][y + dy] == 0) and \
           (0 <= x + dx < grid.shape[0] and 0 <= y - dy < grid.shape[1] and grid[x + dx][y - dy] == 0):
            return node
        if jump(grid, Node(x + dx, y), node, goal) or jump(grid, Node(x, y + dy), node, goal):
            return node

    # Continue jumping in the same direction
    if 0 <= x + dx < grid.shape[0] and 0 <= y + dy < grid.shape[1]:
        return jump(grid, Node(x + dx, y + dy), node, goal)
    
    return None

def identify_successors(grid, node, goal):
    successors = []
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    for direction in neighbors:
        next_node = Node(node.x + direction[0], node.y + direction[1])
        jump_point = jump(grid, next_node, node, goal)
        if jump_point:
            successors.append(jump_point)
    
    return successors

def jps(grid, start, goal):
    open_list = []
    closed_list = set()
    open_dict = {}  # For visualizing nodes in open list
    closed_dict = {}  # For visualizing nodes in closed list

    heapq.heappush(open_list, (start.f, start))
    open_dict[(start.x, start.y)] = True  # Mark node in open list
    
    while open_list:
        _, current_node = heapq.heappop(open_list)
        if (current_node.x, current_node.y) == (goal.x, goal.y):
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1], open_dict, closed_dict  # Return reversed path and visualization data

        closed_list.add((current_node.x, current_node.y))
        closed_dict[(current_node.x, current_node.y)] = True  # Mark node in closed list

        successors = identify_successors(grid, current_node, goal)
        for successor in successors:
            if (successor.x, successor.y) in closed_list:
                continue
            successor.g = current_node.g + 1
            successor.h = heuristic(successor, goal)
            successor.f = successor.g + successor.h

            if (successor.x, successor.y) not in open_dict:
                heapq.heappush(open_list, (successor.f, successor))
                open_dict[(successor.x, successor.y)] = True  # Mark node in open list

        # Adding delay of 0.5 seconds between iterations
        # pygame.time.delay(50)
        visualize_search(open_dict, closed_dict)
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

    # Find path using JPS and get visualization data
    path, open_dict, closed_dict = jps(grid, start, goal)
    current_step = 0
    path_found = False

    while run:
        screen.fill(WHITE)
        draw_grid()
        draw_obstacles(grid)

        # Visualize search process (yellow = open list, orange = closed list)
        if not path_found:
            visualize_search(open_dict, closed_dict)

        # When path is found, draw the path
        if path and not path_found:
            draw_path(path)
            pygame.draw.rect(screen, RED, (goal.y * TILE_SIZE, goal.x * TILE_SIZE, TILE_SIZE, TILE_SIZE))  # Goal

        # Start moving the character once the path is fully drawn
        if path_found:
            draw_path(path)
            pygame.draw.rect(screen, RED, (goal.y * TILE_SIZE, goal.x * TILE_SIZE, TILE_SIZE, TILE_SIZE))  # Goal
            if current_step < len(path):
                # Draw the character moving on the green path
                pygame.draw.rect(screen, BLUE, (path[current_step][1] * TILE_SIZE, path[current_step][0] * TILE_SIZE, TILE_SIZE, TILE_SIZE))
                current_step += 1
            else:
                current_step = 0  # Reset step count when path is completed

        pygame.display.flip()
        clock.tick(10)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

    pygame.quit()

if __name__ == "__main__":
    main()
