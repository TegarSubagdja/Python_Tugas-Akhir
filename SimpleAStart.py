import pygame
import heapq
import numpy as np

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 20, 20
TILE_SIZE = WIDTH // COLS
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Pathfinding with Smoothing")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

def heuristic(node, goal):
    return abs(node.x - goal.x) + abs(node.y - goal.y)

def a_star(grid, start, goal):
    open_list = []
    closed_list = set()
    open_dict = {}
    closed_dict = {}

    heapq.heappush(open_list, (start.f, start))
    open_dict[(start.x, start.y)] = True
    
    while open_list:
        _, current_node = heapq.heappop(open_list)
        if current_node.x == goal.x and current_node.y == goal.y:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1], open_dict, closed_dict

        closed_list.add((current_node.x, current_node.y))
        closed_dict[(current_node.x, current_node.y)] = True

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
                    open_dict[(neighbor_x, neighbor_y)] = True

        pygame.time.delay(50)
        visualize_search(open_dict, closed_dict)
        draw_grid()
        draw_obstacles(grid)
        pygame.display.flip()

    return None, open_dict, closed_dict

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
    grid[5:8, 5] = 1
    grid[7, 6:8] = 1
    grid[10:13, 10:12] = 1
    return grid

def visualize_search(open_dict, closed_dict):
    for (x, y) in open_dict:
        pygame.draw.rect(screen, YELLOW, (y * TILE_SIZE, x * TILE_SIZE, TILE_SIZE, TILE_SIZE))
    for (x, y) in closed_dict:
        pygame.draw.rect(screen, ORANGE, (y * TILE_SIZE, x * TILE_SIZE, TILE_SIZE, TILE_SIZE))

def can_draw_line(grid, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    # Calculate the differences
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    # Determine the number of steps to take
    steps = max(dx, dy)
    x_inc = dx / steps
    y_inc = dy / steps
    
    # Check buffer zone around the line
    for i in range(steps + 1):
        x = int(x1 + i * x_inc)
        y = int(y1 + i * y_inc)
        
        # Check the surrounding cells to create a buffer zone
        for dx in range(-1, 2):  # Check adjacent cells in x
            for dy in range(-1, 2):  # Check adjacent cells in y
                if 0 <= x + dx < grid.shape[0] and 0 <= y + dy < grid.shape[1]:
                    if grid[x + dx][y + dy] == 1:  # if there's an obstacle
                        return False
    return True

def smooth_path(grid, path):
    if not path:
        return []
    
    smoothed_path = [path[0]]
    current_node = path[0]
    
    for i in range(1, len(path) - 1):
        if not can_draw_line(grid, current_node, path[i + 1]):
            smoothed_path.append(path[i])
            current_node = path[i]
    
    smoothed_path.append(path[-1])
    return smoothed_path

def draw_smooth_path(path):
    if path:
        for i in range(len(path) - 1):
            pygame.draw.line(screen, BLUE, (path[i][1] * TILE_SIZE + TILE_SIZE // 2, path[i][0] * TILE_SIZE + TILE_SIZE // 2),
                             (path[i + 1][1] * TILE_SIZE + TILE_SIZE // 2, path[i + 1][0] * TILE_SIZE + TILE_SIZE // 2), 3)

def main():
    run = True
    clock = pygame.time.Clock()

    grid = np.zeros((ROWS, COLS), dtype=int)
    grid = create_obstacles(grid)

    start = Node(0, 0)
    goal = Node(ROWS - 1, COLS - 1)

    path, open_dict, closed_dict = a_star(grid, start, goal)
    smooth_path_result = smooth_path(grid, path) if path else None
    current_step = 0
    path_found = False

    while run:
        screen.fill(WHITE)
        draw_grid()
        draw_obstacles(grid)

        if not path_found:
            visualize_search(open_dict, closed_dict)

        if path and not path_found:
            draw_path(path)
            pygame.draw.rect(screen, RED, (goal.y * TILE_SIZE, goal.x * TILE_SIZE, TILE_SIZE, TILE_SIZE))

        if smooth_path_result and not path_found:
            draw_smooth_path(smooth_path_result)
            pygame.draw.rect(screen, RED, (goal.y * TILE_SIZE, goal.x * TILE_SIZE, TILE_SIZE, TILE_SIZE))

        if path_found:
            draw_smooth_path(smooth_path_result)
            if current_step < len(smooth_path_result):
                pygame.draw.rect(screen, BLUE, (smooth_path_result[current_step][1] * TILE_SIZE, smooth_path_result[current_step][0] * TILE_SIZE, TILE_SIZE, TILE_SIZE))
                current_step += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        pygame.display.flip()

        if not path_found:
            pygame.time.wait(2000)
            path_found = True
        else:
            clock.tick(5)

    pygame.quit()

if __name__ == "__main__":
    main()
